"""
FisherTune-enhanced training script for UniCDR

This script supports multiple experiment configurations:
1. Baseline (standard UniCDR)
2. FIM-only with scheduling
3. DR-FIM
4. Unified parameter selection
5. Shared-only parameter selection
6. Specific-only parameter selection
7. Adaptive (shared frozen, specific tuned)
8. Various perturbation strategies
"""

import argparse
import numpy as np
import torch
import random
from torch.autograd import Variable
from utils.GraphMaker import GraphMaker
from model.fishertune_trainer import FisherTuneTrainer
from model.fishertune import get_experiment_config, FisherTuneConfig
from utils.data import *
import os
import json
import time
import copy
import sys
import codecs

sys.path.insert(1, 'src')


def create_arg_parser():
    """Create argument parser with FisherTune options"""
    parser = argparse.ArgumentParser('UniCDR-FisherTune')

    # DATA Arguments
    parser.add_argument('--domains', type=str, default="sport_cloth",
                        help='Domain names (e.g., sport_cloth, game_video)')
    parser.add_argument('--task', type=str, default='dual-user-intra',
                        help='dual-user-intra, dual-user-inter, multi-item-intra, multi-user-intra')

    # MODEL Arguments
    parser.add_argument('--model', type=str, default='UniCDR', help='Model name')
    parser.add_argument('--mask_rate', type=float, default=0.1, help='Mask rate of interactions')
    parser.add_argument('--num_epoch', type=int, default=100, help='Number of epochs')
    parser.add_argument('--aggregator', type=str, default='mean',
                        help='mean, user_attention, or item_similarity')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size')
    parser.add_argument('--optim', choices=['sgd', 'adagrad', 'adam', 'adamax'], default='adam')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--l2_reg', type=float, default=1e-7, help='L2 weight')
    parser.add_argument('--lr_decay', type=float, default=0.98, help='Learning rate decay')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--latent_dim', type=int, default=128, help='Latent dimensions')
    parser.add_argument('--num_negative', type=int, default=10, help='Negative samples')
    parser.add_argument('--maxlen', type=int, default=10, help='Item sequence length')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--save', action='store_true', help='Save model')
    parser.add_argument('--lambda', type=float, default=50, help='EASE parameter')
    parser.add_argument('--lambda_a', type=float, default=0.5, help='Aggregator weight')
    parser.add_argument('--lambda_loss', type=float, default=0.4, help='Loss weight')
    parser.add_argument('--static_sample', action='store_true', help='Static sampling')

    # FisherTune Arguments
    parser.add_argument('--fishertune_mode', type=str, default='baseline',
                        choices=['baseline', 'fim_only', 'dr_fim', 'unified', 'shared_only',
                                 'specific_only', 'adaptive', 'perturbation_edge_dropout',
                                 'perturbation_popularity', 'perturbation_noise',
                                 'perturbation_cross_domain'],
                        help='FisherTune experiment mode')
    parser.add_argument('--ft_warmup_epochs', type=int, default=5,
                        help='Warmup epochs before applying FisherTune')
    parser.add_argument('--ft_update_freq', type=int, default=5,
                        help='Frequency of Fisher information update')
    parser.add_argument('--ft_num_samples', type=int, default=100,
                        help='Number of samples for Fisher computation')
    parser.add_argument('--ft_delta_min', type=float, default=0.1,
                        help='Minimum threshold for parameter selection')
    parser.add_argument('--ft_delta_max', type=float, default=0.9,
                        help='Maximum threshold for parameter selection')
    parser.add_argument('--ft_schedule_T', type=float, default=10,
                        help='Time constant for exponential scheduling')
    parser.add_argument('--ft_perturbation_rate', type=float, default=0.1,
                        help='Perturbation rate for domain simulation')
    parser.add_argument('--ft_noise_scale', type=float, default=0.01,
                        help='Noise scale for perturbation')

    # Training efficiency
    parser.add_argument('--iterations_per_epoch', type=int, default=100,
                        help='Number of iterations per epoch (default 100, original 500)')

    # Experiment tracking
    parser.add_argument('--experiment_name', type=str, default='',
                        help='Name for this experiment run')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='Directory for experiment logs')

    # Others
    parser.add_argument('--cuda', action='store_true', help='Use CUDA')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--decay_epoch', type=int, default=10, help='Decay epoch')

    return parser


def seed_everything(seed=1111):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_domain_data(opt):
    """Load data for all domains"""
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

    # First pass: get dimensions
    for idx, cur_domain in enumerate(domain_list):
        cur_src_data_dir = os.path.join("../datasets/" + str(opt["task"]) + "/dataset/",
                                        cur_domain + "/train.txt")
        print(f'Loading {cur_domain}: {cur_src_data_dir}')

        all_domain_list.append({})
        all_domain_set.append({})
        max_user = 0
        max_item = 0

        with codecs.open(cur_src_data_dir, "r", encoding="utf-8") as infile:
            for line in infile:
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

    # Build graphs
    total_graphs = GraphMaker(opt, all_domain_list)

    # Second pass: with EASE scores
    all_domain_list = []
    all_domain_set = []

    for idx, cur_domain in enumerate(domain_list):
        cur_src_data_dir = os.path.join("../datasets/" + str(opt["task"]) + "/dataset/",
                                        cur_domain + "/train.txt")

        if opt["aggregator"] == "item_similarity":
            ease_dense = total_graphs.ease[idx].to_dense()

        all_domain_list.append({})
        all_domain_set.append({})

        with codecs.open(cur_src_data_dir, "r", encoding="utf-8") as infile:
            for line in infile:
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

        cur_src_task_generator = TaskGenerator(cur_src_data_dir, opt, all_domain_list,
                                               all_domain_set, idx, total_graphs)
        task_gen_all[idx] = cur_src_task_generator
        domain_id[cur_domain] = idx

    return task_gen_all, domain_id, all_domain_set


def create_fishertune_config(opt):
    """Create FisherTune configuration based on command-line arguments"""
    # Start with preset configuration
    config = get_experiment_config(opt["fishertune_mode"])

    # Override with command-line arguments
    config.warmup_epochs = opt["ft_warmup_epochs"]
    config.fisher_update_freq = opt["ft_update_freq"]
    config.num_samples_fisher = opt["ft_num_samples"]
    config.delta_min = opt["ft_delta_min"]
    config.delta_max = opt["ft_delta_max"]
    config.schedule_T = opt["ft_schedule_T"]
    config.perturbation_rate = opt["ft_perturbation_rate"]
    config.noise_scale = opt["ft_noise_scale"]
    config.device = opt["device"]

    return config


class ExperimentLogger:
    """Logger for tracking experiment results"""

    def __init__(self, log_dir, experiment_name, opt):
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.opt = opt

        os.makedirs(log_dir, exist_ok=True)

        self.results = {
            'config': opt,
            'train_losses': [],
            'val_metrics': {},
            'test_metrics': {},
            'best_metrics': {},
            'fishertune_stats': [],
            'training_time': 0
        }

        # Initialize per-domain tracking
        for domain in opt["domains"]:
            self.results['val_metrics'][domain] = []
            self.results['test_metrics'][domain] = []
            self.results['best_metrics'][domain] = {'NDCG_10': 0, 'HT_10': 0}

    def log_train_loss(self, epoch, avg_loss, domain_losses):
        self.results['train_losses'].append({
            'epoch': epoch,
            'avg_loss': avg_loss,
            'domain_losses': domain_losses
        })

    def log_val_metrics(self, epoch, domain, metrics):
        self.results['val_metrics'][domain].append({
            'epoch': epoch,
            **metrics
        })

    def log_test_metrics(self, epoch, domain, metrics):
        self.results['test_metrics'][domain].append({
            'epoch': epoch,
            **metrics
        })

        # Update best
        if metrics['NDCG_10'] > self.results['best_metrics'][domain]['NDCG_10']:
            self.results['best_metrics'][domain] = metrics

    def log_fishertune_stats(self, stats):
        self.results['fishertune_stats'].append(stats)

    def set_training_time(self, total_time):
        self.results['training_time'] = total_time

    def save(self):
        filename = os.path.join(self.log_dir, f"{self.experiment_name}_results.json")
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"Results saved to {filename}")

    def print_summary(self):
        print("\n" + "=" * 80)
        print(f"EXPERIMENT SUMMARY: {self.experiment_name}")
        print("=" * 80)
        print(f"FisherTune Mode: {self.opt['fishertune_mode']}")
        print(f"Training Time: {self.results['training_time']:.2f} minutes")
        print("\nBest Test Results:")
        for domain, metrics in self.results['best_metrics'].items():
            print(f"  {domain}:")
            print(f"    NDCG@10: {metrics['NDCG_10']:.4f}")
            print(f"    HT@10: {metrics['HT_10']:.4f}")
        print("=" * 80 + "\n")


def main():
    parser = create_arg_parser()
    args = parser.parse_args()
    opt = vars(args)

    opt["device"] = torch.device('cuda' if torch.cuda.is_available() and opt["cuda"] else 'cpu')

    if opt["task"] == "multi-user-intra":
        opt["maxlen"] = 50

    # Print configuration
    print("\n" + "=" * 60)
    print("UNICDR + FISHERTUNE TRAINING")
    print("=" * 60)
    for k, v in opt.items():
        print(f"  {k}: {v}")
    print("=" * 60 + "\n")

    seed_everything(opt["seed"])

    # Setup experiment logging
    if not opt["experiment_name"]:
        opt["experiment_name"] = f"{opt['fishertune_mode']}_{opt['domains']}_{int(time.time())}"

    # Load data (this will modify opt["domains"] from string to list)
    task_gen_all, domain_id, all_domain_set = load_domain_data(opt)

    # Now create logger after domains are properly parsed
    logger = ExperimentLogger(opt["log_dir"], opt["experiment_name"], opt)

    # Create dataloaders
    train_domains = MetaDomain_Dataset(task_gen_all, num_negatives=opt["num_negative"],
                                       meta_split='train')
    train_dataloader = MetaDomain_DataLoader(train_domains,
                                             sample_batch_size=opt["batch_size"] // len(opt["domains"]),
                                             shuffle=True)
    opt["num_domains"] = train_dataloader.num_domains
    opt["domain_id"] = domain_id

    # Validation and test dataloaders
    if "inter" in opt["task"]:
        opt["shared_user"] = int(1e9)

    valid_dataloader = {}
    test_dataloader = {}

    for cur_domain in opt["domains"]:
        if opt["task"] == "dual-user-intra":
            domain_valid = os.path.join("../datasets/" + str(opt["task"]) + "/dataset/",
                                        cur_domain + "/test.txt")
        else:
            domain_valid = os.path.join("../datasets/" + str(opt["task"]) + "/dataset/",
                                        cur_domain + "/valid.txt")
        domain_test = os.path.join("../datasets/" + str(opt["task"]) + "/dataset/",
                                   cur_domain + "/test.txt")

        valid_dataloader[cur_domain] = task_gen_all[domain_id[cur_domain]].instance_a_valid_dataloader(
            domain_valid, 100)
        test_dataloader[cur_domain] = task_gen_all[domain_id[cur_domain]].instance_a_valid_dataloader(
            domain_test, 100)

    print(f"User counts per domain: {opt['user_max']}")
    print(f"Item counts per domain: {opt['item_max']}")

    # Create FisherTune configuration
    ft_config = create_fishertune_config(opt)

    # Pass iteration config to opt
    opt["iterations_per_epoch"] = opt.get("iterations_per_epoch", 100)

    # Initialize model with FisherTune
    mymodel = FisherTuneTrainer(opt, ft_config)

    # Training tracking
    dev_score_history = [[-1] for _ in range(opt["num_domains"])]
    current_lr = opt['lr']
    global_start_time = time.time()

    print(f"\nStarting training for {opt['num_epoch']} epochs...")
    print(f"FisherTune mode: {opt['fishertune_mode']}")

    # Main training loop
    for epoch in range(0, opt["num_epoch"] + 1):
        epoch_start_time = time.time()
        print(f'\nEpoch {epoch} starts!')

        # Skip training for epoch 0
        if epoch == 0:
            avg_loss = 0
            domain_losses = [0] * opt["num_domains"]
        else:
            # Update Fisher information periodically
            if (mymodel.ft_enabled and epoch >= ft_config.warmup_epochs and
                    (epoch - ft_config.warmup_epochs) % ft_config.fisher_update_freq == 0):
                # Create dictionary of dataloaders for Fisher computation
                fisher_dataloader_dict = {}
                for idx in range(opt["num_domains"]):
                    fisher_dataloader_dict[idx] = train_dataloader[idx]
                mymodel.update_fisher_information(fisher_dataloader_dict)

            # Train one epoch
            avg_loss, domain_losses = mymodel.train_epoch(train_dataloader, epoch)

        # Log training loss
        logger.log_train_loss(epoch, avg_loss, domain_losses)

        epoch_time = (time.time() - epoch_start_time) / 60
        print(f"\nAvg loss: {avg_loss:.4f}, Time: {epoch_time:.2f} min, LR: {current_lr:.6f}")

        for idx in range(opt["num_domains"]):
            print(f"  Domain {idx} loss: {domain_losses[idx]:.4f}")

        # Validation every 5 epochs
        if epoch % 5 != 0:
            continue

        print('\nValidation:')
        mymodel.model.eval()
        mymodel.model.item_embedding_select()

        decay_switch = 0
        for idx, cur_domain in enumerate(valid_dataloader):
            if opt["task"] == "multi-user-intra":
                metrics = mymodel.predict_full_rank(idx, valid_dataloader[cur_domain],
                                                    all_domain_set[idx],
                                                    task_gen_all[idx].eval_set)
            else:
                metrics = mymodel.predict(idx, valid_dataloader[cur_domain])

            logger.log_val_metrics(epoch, cur_domain, metrics)

            print(f"\n--- {cur_domain} ---")
            print(f"  NDCG@10: {metrics['NDCG_10']:.4f}")
            print(f"  HT@10: {metrics['HT_10']:.4f}")

            # Check for improvement
            if metrics["NDCG_10"] > max(dev_score_history[idx]):
                print(f"  ** New best for {cur_domain}! **")

                # Test evaluation
                if opt["task"] == "multi-user-intra":
                    test_metrics = mymodel.predict_full_rank(idx, test_dataloader[cur_domain],
                                                             all_domain_set[idx],
                                                             task_gen_all[idx].eval_set)
                else:
                    test_metrics = mymodel.predict(idx, test_dataloader[cur_domain])

                logger.log_test_metrics(epoch, cur_domain, test_metrics)
                print(f"  Test NDCG@10: {test_metrics['NDCG_10']:.4f}")
                print(f"  Test HT@10: {test_metrics['HT_10']:.4f}")

                if opt["save"]:
                    save_path = os.path.join(opt["log_dir"],
                                             f"{opt['experiment_name']}_best_{cur_domain}.pt")
                    mymodel.save(save_path, epoch)
            else:
                decay_switch += 1

            dev_score_history[idx].append(metrics["NDCG_10"])

        # Update warmup flag
        if epoch > opt['decay_epoch']:
            mymodel.model.warmup = 0

        # Learning rate scheduling
        if (epoch > opt['decay_epoch'] and
                decay_switch > opt["num_domains"] // 2 and
                opt['optim'] in ['sgd', 'adagrad', 'adadelta', 'adam']):
            current_lr *= opt['lr_decay']
            mymodel.update_lr(current_lr)
            print(f"\nLearning rate decayed to {current_lr:.6f}")

        # Log FisherTune stats
        if mymodel.ft_enabled:
            stats = mymodel.fishertune.get_statistics()
            logger.log_fishertune_stats(stats)

    # Training complete
    total_training_time = (time.time() - global_start_time) / 60
    logger.set_training_time(total_training_time)

    # Print final summaries
    mymodel.print_fishertune_summary()
    logger.print_summary()
    logger.save()

    print("Experiment finished successfully!")


if __name__ == "__main__":
    main()
