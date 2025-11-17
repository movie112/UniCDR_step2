"""
FisherTune-enhanced training script for UniCDR.

Usage:
    python train_fishertune.py --domains sport_cloth --task dual-user-intra \
        --use_fishertune --fishertune_warmup 10 --shared_lr 0.0001 --specific_lr 0.001
"""

import argparse
import numpy as np
import torch
from torch.autograd import Variable
from utils.GraphMaker import GraphMaker
from fishertune.fishertune_trainer import FisherTuneTrainer
from utils.data import *
import os
import json
import resource
import sys
import pickle
import time
import copy

sys.path.insert(1, 'src')


def create_arg_parser():
    """Create argument parser with FisherTune options."""
    parser = argparse.ArgumentParser('UniCDR-FisherTune')

    # DATA Arguments (same as original)
    parser.add_argument('--domains', type=str, default="sport_cloth",
                        help='domain configuration')
    parser.add_argument('--task', type=str, default='dual-user-intra',
                        help='dual-user-intra, dual-user-inter, multi-item-intra, multi-user-intra')

    # MODEL Arguments (same as original)
    parser.add_argument('--model', type=str, default='UniCDR', help='model name')
    parser.add_argument('--mask_rate', type=float, default=0.1, help='mask rate of interactions')
    parser.add_argument('--num_epoch', type=int, default=100, help='number of epochs')
    parser.add_argument('--aggregator', type=str, default='mean',
                        help='switching the user-item aggregation')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--optim', choices=['sgd', 'adagrad', 'adam', 'adamax'], default='adam',
                        help='Optimizer: sgd, adagrad, adam or adamax.')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--l2_reg', type=float, default=1e-7, help='the L2 weight')
    parser.add_argument('--lr_decay', type=float, default=0.98, help='decay learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay')
    parser.add_argument('--latent_dim', type=int, default=128, help='latent dimensions')
    parser.add_argument('--num_negative', type=int, default=10,
                        help='num of negative samples during training')
    parser.add_argument('--maxlen', type=int, default=10, help='num of item sequence')
    parser.add_argument('--dropout', type=float, default=0.3, help='random drop out rate')
    parser.add_argument('--save', action='store_true', help='save model?')
    parser.add_argument('--lambda', type=float, default=50, help='the parameter of EASE')
    parser.add_argument('--lambda_a', type=float, default=0.5, help='for our aggregators')
    parser.add_argument('--lambda_loss', type=float, default=0.4, help='parameter of loss function')
    parser.add_argument('--static_sample', action='store_true', help='accelerate the dataloader')

    # FisherTune Arguments
    parser.add_argument('--use_fishertune', action='store_true',
                        help='Enable FisherTune optimization')
    parser.add_argument('--fishertune_warmup', type=int, default=10,
                        help='Warmup epochs before applying FisherTune')
    parser.add_argument('--fim_update_freq', type=int, default=10,
                        help='Frequency of FIM updates (epochs)')

    # Shared vs Specific Parameter Thresholds
    parser.add_argument('--shared_delta_min', type=float, default=0.5,
                        help='Min threshold for shared params (high = more frozen)')
    parser.add_argument('--shared_delta_max', type=float, default=0.95,
                        help='Max threshold for shared params')
    parser.add_argument('--specific_delta_min', type=float, default=0.1,
                        help='Min threshold for specific params (low = more tuning)')
    parser.add_argument('--specific_delta_max', type=float, default=0.7,
                        help='Max threshold for specific params')
    parser.add_argument('--threshold_decay_constant', type=float, default=50.0,
                        help='Time constant for threshold decay (T)')

    # Learning Rate Configuration
    parser.add_argument('--shared_lr', type=float, default=0.0001,
                        help='Learning rate for shared parameters')
    parser.add_argument('--specific_lr', type=float, default=0.001,
                        help='Learning rate for domain-specific parameters')
    parser.add_argument('--shared_weight_decay', type=float, default=1e-3,
                        help='Weight decay for shared parameters')
    parser.add_argument('--specific_weight_decay', type=float, default=1e-5,
                        help='Weight decay for specific parameters')

    # Perturbation Settings
    parser.add_argument('--perturbation_type', type=str, default='combined',
                        choices=['edge_dropout', 'popularity', 'noise', 'mask', 'combined'],
                        help='Type of domain perturbation')
    parser.add_argument('--edge_dropout_rate', type=float, default=0.2,
                        help='Edge dropout rate for perturbation')
    parser.add_argument('--perturbation_noise_std', type=float, default=0.1,
                        help='Noise std for perturbation')
    parser.add_argument('--popularity_alpha', type=float, default=0.5,
                        help='Weight for popularity-based perturbation')
    parser.add_argument('--embedding_noise_std', type=float, default=0.1,
                        help='Noise std for embedding perturbation')
    parser.add_argument('--perturbation_noise', type=float, default=0.1,
                        help='Combined perturbation magnitude for DR-FIM weighting')

    # Variational Inference Settings
    parser.add_argument('--variational_tau', type=float, default=1.0,
                        help='Prior std for variational inference')
    parser.add_argument('--variational_gamma', type=float, default=0.1,
                        help='KL divergence weight')
    parser.add_argument('--use_adaptive_vi', action='store_true',
                        help='Use adaptive variational inference')
    parser.add_argument('--tau_decay', type=float, default=0.99,
                        help='Tau decay rate for adaptive VI')
    parser.add_argument('--gamma_growth', type=float, default=1.01,
                        help='Gamma growth rate for adaptive VI')

    # Fisher Estimation Settings
    parser.add_argument('--fisher_momentum', type=float, default=0.9,
                        help='Momentum for online Fisher estimation')
    parser.add_argument('--fisher_damping', type=float, default=1e-4,
                        help='Damping for Fisher estimation')
    parser.add_argument('--fim_method', type=str, default='online',
                        choices=['online', 'batch'],
                        help='FIM computation method')

    # Advanced Settings
    parser.add_argument('--use_natural_gradient', action='store_true',
                        help='Use natural gradient updates')
    parser.add_argument('--use_fisher_regularization', action='store_true',
                        help='Use Fisher-based regularization')
    parser.add_argument('--fisher_reg_strength', type=float, default=0.1,
                        help='Strength of Fisher regularization')
    parser.add_argument('--use_variational_loss', action='store_true',
                        help='Add variational KL loss')
    parser.add_argument('--use_adaptive_scheduler', action='store_true',
                        help='Use adaptive threshold scheduler')
    parser.add_argument('--scheduler_adaptation_rate', type=float, default=0.1,
                        help='Adaptation rate for scheduler')
    parser.add_argument('--percentile_based_selection', action='store_true', default=True,
                        help='Use percentile-based parameter selection')

    # Others (same as original)
    parser.add_argument('--cuda', action='store_true', help='use of cuda')
    parser.add_argument('--seed', type=int, default=42, help='manual seed init')
    parser.add_argument('--decay_epoch', type=int, default=10,
                        help='Decay learning rate after this epoch.')
    parser.add_argument('--exp_name', type=str, default='fishertune',
                        help='Experiment name for logging')

    return parser


def main():
    parser = create_arg_parser()
    args = parser.parse_args()

    opt = vars(args)

    opt["device"] = torch.device(
        'cuda' if torch.cuda.is_available() and opt["cuda"] else 'cpu'
    )

    def print_config(config):
        info = "Running with the following configs:\n"
        for k, v in config.items():
            info += "\t{} : {}\n".format(k, str(v))
        print("\n" + info + "\n")

    if opt["task"] == "multi-user-intra":
        opt["maxlen"] = 50

    print_config(opt)

    print(f'Running FisherTune experiment on device: {opt["device"]}')
    print(f'FisherTune enabled: {opt["use_fishertune"]}')

    def seed_everything(seed=1111):
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    seed_everything(opt["seed"])

    ############
    ## All Domains Data
    ############

    if "dual" in opt["task"]:
        filename = opt["domains"].split("_")
        opt["domains"] = []
        opt["domains"].append(filename[0] + "_" + filename[1])
        opt["domains"].append(filename[1] + "_" + filename[0])
    else:
        opt["domains"] = opt["domains"].split('_')

    print("Loading domains:", opt["domains"])

    domain_list = opt["domains"]
    opt["user_max"] = []
    opt["item_max"] = []
    task_gen_all = {}
    domain_id = {}

    all_domain_list = []
    all_domain_set = []
    all_inter = 0

    for idx, cur_domain in enumerate(domain_list):
        cur_src_data_dir = os.path.join(
            "../datasets/" + str(opt["task"]) + "/dataset/",
            cur_domain + "/train.txt"
        )
        print(f'Loading {cur_domain}: {cur_src_data_dir}')

        all_domain_list.append({})
        all_domain_set.append({})
        max_user = 0
        max_item = 0

        with codecs.open(cur_src_data_dir, "r", encoding="utf-8") as infile:
            for line in infile:
                all_inter += 1
                line = line.strip().split("\t")
                user = int(line[0])
                item = int(line[1]) + 1
                max_user = max(max_user, user)
                max_item = max(max_item, item)
                if user not in all_domain_list[idx].keys():
                    all_domain_list[idx][user] = []
                    all_domain_set[idx][user] = set()
                if item not in all_domain_set[idx][user]:
                    all_domain_list[idx][user].append(item)
                    all_domain_set[idx][user].add(item)

        opt["user_max"].append(max_user + 1)
        opt["item_max"].append(max_item + 1)

    total_graphs = GraphMaker(opt, all_domain_list)

    # Reload with EASE scores if needed
    all_domain_list = []
    all_domain_set = []
    all_inter = 0

    for idx, cur_domain in enumerate(domain_list):
        cur_src_data_dir = os.path.join(
            "../datasets/" + str(opt["task"]) + "/dataset/",
            cur_domain + "/train.txt"
        )
        print(f'Loading {cur_domain}: {cur_src_data_dir}')

        if opt["aggregator"] == "item_similarity":
            ease_dense = total_graphs.ease[idx].to_dense()

        all_domain_list.append({})
        all_domain_set.append({})

        with codecs.open(cur_src_data_dir, "r", encoding="utf-8") as infile:
            for line in infile:
                all_inter += 1
                line = line.strip().split("\t")
                user = int(line[0])
                item = int(line[1]) + 1
                if user not in all_domain_list[idx].keys():
                    all_domain_list[idx][user] = []
                    all_domain_set[idx][user] = set()
                if item not in all_domain_set[idx][user]:
                    if opt["aggregator"] == "item_similarity":
                        all_domain_list[idx][user].append([item, ease_dense[user][item]])
                    else:
                        all_domain_list[idx][user].append([item, 1])
                    all_domain_set[idx][user].add(item)

        print(f'Loading {cur_domain}: {cur_src_data_dir}')
        cur_src_task_generator = TaskGenerator(
            cur_src_data_dir, opt, all_domain_list, all_domain_set,
            idx, total_graphs
        )
        task_gen_all[idx] = cur_src_task_generator
        domain_id[cur_domain] = idx

    train_domains = MetaDomain_Dataset(
        task_gen_all, num_negatives=opt["num_negative"], meta_split='train'
    )
    train_dataloader = MetaDomain_DataLoader(
        train_domains, sample_batch_size=opt["batch_size"] // len(domain_list),
        shuffle=True
    )
    opt["num_domains"] = train_dataloader.num_domains
    opt["num_domain"] = train_dataloader.num_domains
    opt["domain_id"] = domain_id

    ############
    ## Validation and Test
    ############
    if "inter" in opt["task"]:
        opt["shared_user"] = 1e9

    valid_dataloader = {}
    test_dataloader = {}
    for cur_domain in domain_list:
        if opt["task"] == "dual-user-intra":
            domain_valid = os.path.join(
                "../datasets/" + str(opt["task"]) + "/dataset/",
                cur_domain + "/test.txt"
            )
        else:
            domain_valid = os.path.join(
                "../datasets/" + str(opt["task"]) + "/dataset/",
                cur_domain + "/valid.txt"
            )
        domain_test = os.path.join(
            "../datasets/" + str(opt["task"]) + "/dataset/",
            cur_domain + "/test.txt"
        )
        valid_dataloader[cur_domain] = task_gen_all[domain_id[cur_domain]].instance_a_valid_dataloader(
            domain_valid, 100
        )
        test_dataloader[cur_domain] = task_gen_all[domain_id[cur_domain]].instance_a_valid_dataloader(
            domain_test, 100
        )

    print("the user number of different domains", opt["user_max"])
    print("the item number of different domains", opt["item_max"])

    ############
    ## Model with FisherTune
    ############
    print("\n" + "=" * 50)
    print("Initializing FisherTune Trainer")
    print("=" * 50)

    mymodel = FisherTuneTrainer(opt)

    # Set domain data for FIM computation
    if opt["use_fishertune"]:
        # Convert train_dataloader to list format for FIM computation
        domain_loaders_list = [train_dataloader[i] for i in range(opt["num_domains"])]
        domain_interactions = {
            i: all_domain_list[i] for i in range(len(all_domain_list))
        }
        mymodel.set_domain_data(
            domain_loaders_list,
            domain_interactions,
            opt["item_max"]
        )

    ############
    ## Train
    ############
    dev_score_history = []
    for i in range(opt["num_domains"]):
        dev_score_history.append([0])

    current_lr = opt['lr']
    iteration_num = 500

    print("per batch of an epoch:", iteration_num)
    global_step = 0

    # Logging
    training_log = {
        'losses': [],
        'metrics': [],
        'fishertune_stats': []
    }

    for epoch in range(0, opt["num_epoch"] + 1):
        start_time = time.time()
        print('=' * 80)
        print(f'Epoch {epoch} starts !')

        # FisherTune: epoch start hook
        if opt["use_fishertune"]:
            mymodel.on_epoch_start(epoch)

        total_loss = [0]
        loss_list = []
        for i in range(opt["num_domains"]):
            loss_list.append([0])

        for iteration in range(iteration_num):
            if epoch == 0:
                continue
            if iteration % 10 == 0:
                print(".", end="")

            mymodel.model.train()
            mymodel.model.item_embedding_select()

            if opt["use_fishertune"] and epoch >= opt["fishertune_warmup"]:
                # Use FisherTune training step
                mymodel.ft_optimizer.zero_grad()
                mymodel_loss = 0

                for idx in range(opt["num_domains"]):
                    global_step += 1
                    cur_train_dataloader = train_dataloader.get_iterator(idx)
                    try:
                        batch_data = next(cur_train_dataloader)
                    except:
                        new_train_iterator = iter(train_dataloader[idx])
                        batch_data = next(new_train_iterator)

                    cur_loss = mymodel.reconstruct_graph(idx, batch_data)
                    mymodel_loss += cur_loss
                    loss_list[idx].append(cur_loss.item())
                    total_loss.append(cur_loss.item())

                mymodel_loss.backward()

                # Update online Fisher estimate
                with torch.no_grad():
                    mymodel.online_fisher.update(mymodel_loss.detach())

                # Apply parameter selection
                mymodel._apply_parameter_selection()

                mymodel.ft_optimizer.step()
                mymodel.training_step += 1

            else:
                # Standard training (warmup phase)
                mymodel.optimizer.zero_grad()
                mymodel_loss = 0

                for idx in range(opt["num_domains"]):
                    global_step += 1
                    cur_train_dataloader = train_dataloader.get_iterator(idx)
                    try:
                        batch_data = next(cur_train_dataloader)
                    except:
                        new_train_iterator = iter(train_dataloader[idx])
                        batch_data = next(new_train_iterator)

                    cur_loss = mymodel.reconstruct_graph(idx, batch_data)
                    mymodel_loss += cur_loss
                    loss_list[idx].append(cur_loss.item())
                    total_loss.append(cur_loss.item())

                mymodel_loss.backward()
                mymodel.optimizer.step()

        avg_loss = sum(total_loss) / len(total_loss)
        epoch_time = (time.time() - start_time) / 60

        print(f"\nAverage loss: {avg_loss:.6f}, time: {epoch_time:.2f} (min), "
              f"current lr: {current_lr:.6f}")

        # Log training info
        training_log['losses'].append({
            'epoch': epoch,
            'avg_loss': avg_loss,
            'domain_losses': [sum(l) / len(l) if len(l) > 1 else 0 for l in loss_list]
        })

        # FisherTune: epoch end hook
        if opt["use_fishertune"]:
            mymodel.epoch_rec_loss = total_loss
            mymodel.on_epoch_end(epoch)

            # Log FisherTune stats
            ft_stats = mymodel.get_training_stats()
            training_log['fishertune_stats'].append({
                'epoch': epoch,
                **ft_stats
            })

        print('-' * 80)

        if epoch % 5:
            continue

        for idx in range(opt["num_domains"]):
            print(f"Domain {idx} loss: {sum(loss_list[idx]) / len(loss_list[idx]):.6f}")

        print('Make prediction:')
        valid_start = time.time()

        mymodel.model.eval()
        mymodel.model.item_embedding_select()

        decay_switch = 0
        epoch_metrics = {}

        for idx, cur_domain in enumerate(valid_dataloader):
            if opt["task"] == "multi-user-intra":
                metrics = mymodel.predict_full_rank(
                    idx, valid_dataloader[cur_domain],
                    all_domain_set[idx], task_gen_all[idx].eval_set
                )
            else:
                metrics = mymodel.predict(idx, valid_dataloader[cur_domain])

            print(f"\n-------------------{cur_domain}--------------------")
            print(metrics)

            epoch_metrics[cur_domain] = metrics

            if metrics["NDCG_10"] > max(dev_score_history[idx]):
                print(f"{cur_domain} better results!")

                if opt["save"]:
                    save_path = f"saved_models/{opt['exp_name']}_{cur_domain}_best.pt"
                    os.makedirs("saved_models", exist_ok=True)
                    mymodel.save(save_path, epoch)
                    print(f"best model saved to {save_path}!")

                if opt["task"] == "multi-user-intra":
                    test_metrics = mymodel.predict_full_rank(
                        idx, test_dataloader[cur_domain],
                        all_domain_set[idx], task_gen_all[idx].eval_set
                    )
                else:
                    test_metrics = mymodel.predict(idx, test_dataloader[cur_domain])

                print("Test metrics:", test_metrics)
                epoch_metrics[f"{cur_domain}_test"] = test_metrics
            else:
                decay_switch += 1

            dev_score_history[idx].append(metrics["NDCG_10"])

        training_log['metrics'].append({
            'epoch': epoch,
            **epoch_metrics
        })

        print(f"valid time: {(time.time() - valid_start) / 60:.2f} (min)")

        if epoch > opt['decay_epoch']:
            mymodel.model.warmup = 0

        # lr schedule
        print("decay_switch:", decay_switch)
        if ((epoch > opt['decay_epoch']) and
            (decay_switch > opt["num_domains"] // 2) and
            (opt['optim'] in ['sgd', 'adagrad', 'adadelta', 'adam'])):
            current_lr *= opt['lr_decay']
            mymodel.update_lr(current_lr)

    # Save training log
    log_path = f"logs/{opt['exp_name']}_training_log.json"
    os.makedirs("logs", exist_ok=True)
    with open(log_path, 'w') as f:
        # Convert non-serializable items
        def convert_to_serializable(obj):
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, set):
                return list(obj)
            else:
                return obj

        json.dump(convert_to_serializable(training_log), f, indent=2)

    print(f"\nTraining log saved to {log_path}")
    print('Experiment finished successfully!')


if __name__ == "__main__":
    main()
