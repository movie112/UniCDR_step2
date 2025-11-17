"""
FisherTune-enhanced trainer for UniCDR
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from model.UniCDR import UniCDR
from model.fishertune import FisherTuneModule, DomainPerturbation, FisherTuneConfig
from model.fishertune_efficient import EfficientFisherComputer, AdaptiveScheduler, print_efficiency_report
from utils import torch_utils
import numpy as np
import time
import math


class FisherTuneTrainer:
    """
    Enhanced trainer that incorporates FisherTune methodology
    """

    def __init__(self, opt, fishertune_config=None):
        self.opt = opt

        # Initialize base model
        if self.opt["model"] == "UniCDR":
            self.model = UniCDR(opt)
        else:
            print("please input right model name!")
            exit(0)

        self.criterion = nn.BCEWithLogitsLoss()
        if opt['cuda']:
            self.model.cuda()
            self.criterion.cuda()

        self.optimizer = torch_utils.get_optimizer(
            opt['optim'], self.model.parameters(), opt['lr'], opt["weight_decay"]
        )

        # FisherTune components
        if fishertune_config is None:
            fishertune_config = FisherTuneConfig(device=opt["device"])
        self.ft_config = fishertune_config

        # Initialize FisherTune if enabled
        if self.ft_config.use_fim or self.ft_config.use_dr_fim:
            self.fishertune = FisherTuneModule(self.model, self.ft_config)
            self.perturbation = DomainPerturbation(self.ft_config, opt)
            self.efficient_fisher = EfficientFisherComputer(self.model, self.ft_config.device)
            self.adaptive_scheduler = AdaptiveScheduler(initial_ratio=0.3, target_ratio=0.7)
            self.ft_enabled = True

            # Print efficiency report
            print_efficiency_report(self.ft_config, self.model, opt["num_domains"])
        else:
            self.fishertune = None
            self.perturbation = None
            self.efficient_fisher = None
            self.adaptive_scheduler = None
            self.ft_enabled = False

        # Training statistics
        self.epoch_rec_loss = []
        self.ft_stats_history = []
        self.current_epoch = 0

        # Cache for Fisher computation dataloaders
        self.fisher_dataloader_cache = None
        self.perturbed_dataloader_cache = None

        # Use online Fisher updates for efficiency
        self.use_online_fisher = True

    def unpack_batch_predict(self, batch):
        if self.opt["cuda"]:
            inputs = [Variable(b.cuda()) for b in batch]
        else:
            inputs = [Variable(b) for b in batch]

        return (inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5])

    def unpack_batch(self, batch):
        if self.opt["cuda"]:
            inputs = [Variable(b.cuda()) for b in batch]
        else:
            inputs = [Variable(b) for b in batch]

        return (inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], inputs[6])

    def update_lr(self, new_lr):
        torch_utils.change_lr(self.optimizer, new_lr)

    def save(self, filename, epoch=None):
        params = {
            'model': self.model.state_dict(),
            'config': self.opt,
            'fishertune_enabled': self.ft_enabled,
        }
        if self.ft_enabled and self.fishertune is not None:
            params['fisher_stats'] = self.fishertune.get_statistics()
            params['ft_stats_history'] = self.ft_stats_history

        try:
            torch.save(params, filename)
            print(f"Model saved to {filename}")
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")

    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print(f"Cannot load model from {filename}")
            exit()
        self.model.load_state_dict(checkpoint['model'])
        self.opt = checkpoint['config']

    def reconstruct_graph(self, domain_id, batch):
        """Compute reconstruction loss for a domain"""
        user, pos_item, neg_item, context_item, context_score, global_item, global_score = self.unpack_batch(batch)

        user_feature = self.model.forward_user(domain_id, user, context_item, context_score,
                                               global_item, global_score)
        pos_item_feature = self.model.forward_item(domain_id, pos_item)
        neg_item_feature = self.model.forward_item(domain_id, neg_item)

        pos_score = self.model.predict_dot(user_feature, pos_item_feature)
        neg_score = self.model.predict_dot(user_feature, neg_item_feature)

        pos_labels, neg_labels = torch.ones(pos_score.size()), torch.zeros(neg_score.size())

        if self.opt["cuda"]:
            pos_labels = pos_labels.cuda()
            neg_labels = neg_labels.cuda()

        loss = self.opt["lambda_loss"] * (
            self.criterion(pos_score, pos_labels) + self.criterion(neg_score, neg_labels)
        ) + (1 - self.opt["lambda_loss"]) * self.model.critic_loss

        return loss

    def update_fisher_information(self, train_dataloader_dict):
        """
        Update Fisher information estimates

        This should be called periodically during training (not every epoch to save time)
        """
        if not self.ft_enabled:
            return

        print("\n--- Updating Fisher Information ---")
        start_time = time.time()

        # Prepare dataloaders for Fisher computation
        fisher_dataloaders = {}
        for domain_id in range(self.opt["num_domains"]):
            # Create a small subset for efficient Fisher computation
            fisher_dataloaders[domain_id] = train_dataloader_dict[domain_id]

        if self.ft_config.use_variational:
            self.fishertune.compute_variational_fisher(
                fisher_dataloaders, self.criterion, self.ft_config.num_samples_fisher
            )

        if self.ft_config.use_dr_fim:
            # Create perturbed dataloaders
            perturbed_dataloaders = self.perturbation.create_perturbed_dataloader(fisher_dataloaders)
            self.fishertune.compute_dr_fim(
                fisher_dataloaders, perturbed_dataloaders, self.criterion,
                self.ft_config.num_samples_fisher
            )
        elif self.ft_config.use_fim and not self.ft_config.use_variational:
            self.fishertune.compute_fisher_information(
                fisher_dataloaders, self.criterion, self.ft_config.num_samples_fisher
            )

        print(f"Fisher update completed in {time.time() - start_time:.2f}s")

    def train_epoch(self, train_dataloader, epoch):
        """
        Train one epoch with FisherTune
        """
        self.current_epoch = epoch

        # Update FisherTune masks if enabled
        if self.ft_enabled and epoch >= self.ft_config.warmup_epochs:
            self.fishertune.update_parameter_masks(epoch)

        total_loss = []
        loss_list = [[] for _ in range(self.opt["num_domains"])]

        iteration_num = 500  # Fixed number of iterations per epoch
        self.model.train()

        for iteration in range(iteration_num):
            if iteration % 50 == 0:
                print(".", end="", flush=True)

            self.optimizer.zero_grad()
            self.model.item_embedding_select()
            mymodel_loss = 0

            for domain_id in range(self.opt["num_domains"]):
                cur_train_dataloader = train_dataloader.get_iterator(domain_id)
                try:
                    batch_data = next(cur_train_dataloader)
                except:
                    new_train_iterator = iter(train_dataloader[domain_id])
                    batch_data = next(new_train_iterator)

                cur_loss = self.reconstruct_graph(domain_id, batch_data)
                mymodel_loss += cur_loss
                loss_list[domain_id].append(cur_loss.item())
                total_loss.append(cur_loss.item())

            mymodel_loss.backward()

            # Online Fisher update (efficient - no extra computation)
            if self.ft_enabled and self.use_online_fisher and epoch >= self.ft_config.warmup_epochs:
                self.efficient_fisher.update_fisher_online(mymodel_loss)

            # Apply FisherTune gradient masking
            if self.ft_enabled and epoch >= self.ft_config.warmup_epochs:
                self.fishertune.apply_gradient_mask()

            self.optimizer.step()

        avg_loss = sum(total_loss) / len(total_loss) if total_loss else 0
        domain_losses = [sum(l) / len(l) if l else 0 for l in loss_list]

        # Update adaptive scheduler if enabled
        if self.ft_enabled and self.adaptive_scheduler:
            self.adaptive_scheduler.update_schedule(avg_loss)

        # Record FisherTune statistics
        if self.ft_enabled:
            stats = self.fishertune.get_statistics()
            stats['epoch'] = epoch
            stats['avg_loss'] = avg_loss
            stats['adaptive_ratio'] = self.adaptive_scheduler.current_ratio if self.adaptive_scheduler else 0
            self.ft_stats_history.append(stats)

        return avg_loss, domain_losses

    def predict(self, domain_id, eval_dataloader):
        """Evaluate on validation/test set"""
        MRR = 0.0
        NDCG_1 = 0.0
        NDCG_5 = 0.0
        NDCG_10 = 0.0
        HT_1 = 0.0
        HT_5 = 0.0
        HT_10 = 0.0
        valid_entity = 0

        for test_batch in eval_dataloader:
            user, item, context_item, context_score, global_item, global_score = self.unpack_batch_predict(test_batch)

            user_feature = self.model.forward_user(domain_id, user, context_item, context_score,
                                                   global_item, global_score)
            item_feature = self.model.forward_item(domain_id, item)

            scores = self.model.predict_dot(user_feature, item_feature)
            scores = scores.data.detach().cpu().numpy()

            for pred in scores:
                rank = (-pred).argsort().argsort()[0].item()

                valid_entity += 1
                MRR += 1 / (rank + 1)
                if rank < 1:
                    NDCG_1 += 1 / np.log2(rank + 2)
                    HT_1 += 1
                if rank < 5:
                    NDCG_5 += 1 / np.log2(rank + 2)
                    HT_5 += 1
                if rank < 10:
                    NDCG_10 += 1 / np.log2(rank + 2)
                    HT_10 += 1
                if valid_entity % 100 == 0:
                    print('+', end='')

        print("")
        metrics = {}
        metrics["NDCG_10"] = NDCG_10 / valid_entity
        metrics["HT_10"] = HT_10 / valid_entity
        metrics["MRR"] = MRR / valid_entity

        return metrics

    def predict_full_rank(self, domain_id, eval_dataloader, train_map, eval_map):
        """Full-rank evaluation for multi-user-intra task"""

        def nDCG(ranked_list, ground_truth_length):
            dcg = 0
            idcg = IDCG(ground_truth_length)
            for i in range(len(ranked_list)):
                if ranked_list[i]:
                    rank = i + 1
                    dcg += 1 / math.log(rank + 1, 2)
            return dcg / idcg

        def IDCG(n):
            idcg = 0
            for i in range(n):
                idcg += 1 / math.log(i + 2, 2)
            return idcg

        ndcg_list = []
        pre_list = []
        rec_list = []
        NDCG_10 = 0.0
        HT_10 = 0

        for test_batch in eval_dataloader:
            user, item, context_item, context_score, global_item, global_score = self.unpack_batch_predict(test_batch)

            user_feature = self.model.forward_user(domain_id, user, context_item, context_score,
                                                   global_item, global_score)
            item_feature = self.model.forward_item(domain_id, item)

            scores = self.model.predict_dot(user_feature, item_feature)
            scores = scores.data.detach().cpu().numpy()
            user = user.data.detach().cpu().numpy()

            for idx, pred in enumerate(scores):
                rank = (-pred).argsort()
                score_list = []
                hr = 0

                for i in rank:
                    i = i + 1
                    if (i in train_map[user[idx]]) and (i not in eval_map[user[idx]]):
                        continue
                    else:
                        if i in eval_map[user[idx]]:
                            hr = 1
                            score_list.append(1)
                        else:
                            score_list.append(0)
                        if len(score_list) == 10:
                            break

                HT_10 += hr

                def precision_and_recall(ranked_list, ground_number):
                    hits = sum(ranked_list)
                    pre = hits / (1.0 * len(ranked_list))
                    rec = hits / (1.0 * ground_number)
                    return pre, rec

                pre, rec = precision_and_recall(score_list, len(eval_map[user[idx]]))
                pre_list.append(pre)
                rec_list.append(rec)
                ndcg_list.append(nDCG(score_list, len(eval_map[user[idx]])))

                if len(ndcg_list) % 100 == 0:
                    print('+', end='')

        print("")
        metrics = {}
        metrics["HT_10"] = HT_10 / len(ndcg_list)
        metrics["NDCG_10"] = sum(ndcg_list) / len(ndcg_list)
        metrics["MRR"] = 0

        return metrics

    def print_fishertune_summary(self):
        """Print summary of FisherTune statistics"""
        if not self.ft_enabled or not self.ft_stats_history:
            print("No FisherTune statistics available")
            return

        print("\n" + "=" * 60)
        print("FISHERTUNE SUMMARY")
        print("=" * 60)

        latest_stats = self.ft_stats_history[-1]
        print(f"Mode: {self.ft_config.param_mode}")
        print(f"DR-FIM: {self.ft_config.use_dr_fim}")
        print(f"Perturbation: {self.ft_config.perturbation_type}")
        print(f"\nLatest Fisher Statistics (Epoch {latest_stats['epoch']}):")
        print(f"  Avg Shared FIM: {latest_stats['avg_shared_fim']:.6f}")
        print(f"  Avg Specific FIM: {latest_stats['avg_specific_fim']:.6f}")
        print(f"  Shared Selection Ratio: {latest_stats['shared_selection_ratio']:.2%}")
        print(f"  Specific Selection Ratio: {latest_stats['specific_selection_ratio']:.2%}")
        print("=" * 60 + "\n")
