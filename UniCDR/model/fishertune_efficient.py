"""
Efficient Fisher Information computation for UniCDR

This module provides optimized methods for computing Fisher Information
with minimal computational overhead.
"""

import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
import time


class EfficientFisherComputer:
    """
    Efficient computation of Fisher Information using:
    1. Gradient accumulation (no per-sample computation)
    2. Moving average updates (incremental computation)
    3. Block-wise computation (for large models)
    4. Sparse update patterns
    """

    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device

        # Storage for running Fisher estimates
        self.running_fim = {}
        self.fim_count = {}

        # Initialize
        for name, param in model.named_parameters():
            self.running_fim[name] = torch.zeros_like(param.data)
            self.fim_count[name] = 0

        # Exponential moving average coefficient
        self.ema_decay = 0.99

    def update_fisher_online(self, loss):
        """
        Update Fisher information online during training (O(1) overhead per batch)

        This is called after loss.backward() but before optimizer.step()
        No additional forward/backward passes needed!
        """
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                # Squared gradient is the diagonal Fisher
                grad_sq = param.grad.data ** 2

                # Exponential moving average for smooth updates
                if self.fim_count[name] == 0:
                    self.running_fim[name] = grad_sq
                else:
                    self.running_fim[name] = (self.ema_decay * self.running_fim[name] +
                                              (1 - self.ema_decay) * grad_sq)

                self.fim_count[name] += 1

    def get_fisher(self):
        """Get current Fisher information estimates"""
        return self.running_fim

    def reset(self):
        """Reset Fisher estimates"""
        for name in self.running_fim:
            self.running_fim[name].zero_()
            self.fim_count[name] = 0


class AdaptiveScheduler:
    """
    Adaptive scheduling for FisherTune parameter selection

    Automatically adjusts thresholds based on training dynamics
    """

    def __init__(self, initial_ratio=0.3, target_ratio=0.7, adaptation_rate=0.1):
        """
        Args:
            initial_ratio: Initial fraction of parameters to update
            target_ratio: Target fraction of parameters to update
            adaptation_rate: How quickly to adapt
        """
        self.current_ratio = initial_ratio
        self.target_ratio = target_ratio
        self.adaptation_rate = adaptation_rate

        # Track performance
        self.prev_loss = float('inf')
        self.loss_history = []

    def compute_threshold(self, fim_values, param_type='unified'):
        """
        Compute threshold that selects approximately target_ratio of parameters

        Args:
            fim_values: Dictionary of Fisher information values
            param_type: 'unified', 'shared', or 'specific'

        Returns:
            threshold value
        """
        # Flatten all FIM values
        all_values = []
        for name, fim in fim_values.items():
            all_values.extend(fim.flatten().cpu().numpy())

        if len(all_values) == 0:
            return 0.0

        # Compute percentile threshold
        percentile = (1 - self.current_ratio) * 100
        threshold = np.percentile(all_values, percentile)

        return threshold

    def update_schedule(self, current_loss):
        """
        Update the target ratio based on training progress

        If loss is improving, gradually increase parameter updates
        If loss is stagnating, be more conservative
        """
        self.loss_history.append(current_loss)

        if len(self.loss_history) > 1:
            loss_change = self.prev_loss - current_loss

            if loss_change > 0:
                # Loss improving - can update more parameters
                self.current_ratio = min(self.target_ratio,
                                        self.current_ratio + self.adaptation_rate)
            else:
                # Loss not improving - be conservative
                self.current_ratio = max(0.1,
                                        self.current_ratio - self.adaptation_rate * 0.5)

        self.prev_loss = current_loss

        return self.current_ratio


class GradientCaching:
    """
    Cache gradients for efficient Fisher computation across multiple batches
    """

    def __init__(self, cache_size=10):
        self.cache_size = cache_size
        self.gradient_cache = defaultdict(list)

    def add_gradient(self, name, grad):
        """Add gradient to cache"""
        if len(self.gradient_cache[name]) >= self.cache_size:
            self.gradient_cache[name].pop(0)
        self.gradient_cache[name].append(grad.clone())

    def compute_fisher_from_cache(self):
        """Compute Fisher from cached gradients"""
        fim = {}
        for name, grads in self.gradient_cache.items():
            if len(grads) > 0:
                # Stack gradients
                stacked = torch.stack(grads)
                # Compute variance = E[g^2] - E[g]^2
                # For Fisher, we want E[g^2]
                fim[name] = (stacked ** 2).mean(dim=0)
        return fim


class BlockWiseFisher:
    """
    Compute Fisher information block-wise for memory efficiency

    Particularly useful for large embedding tables
    """

    def __init__(self, model, block_size=10000):
        self.model = model
        self.block_size = block_size

    def compute_embedding_fisher(self, embedding_module, used_indices):
        """
        Compute Fisher only for used embedding indices

        Args:
            embedding_module: nn.Embedding layer
            used_indices: Set of indices that were used in training

        Returns:
            Sparse Fisher information (only for used indices)
        """
        device = embedding_module.weight.device
        num_embeddings = embedding_module.num_embeddings
        embedding_dim = embedding_module.embedding_dim

        # Only compute for used indices
        used_indices = sorted(list(used_indices))
        if len(used_indices) == 0:
            return torch.zeros(num_embeddings, embedding_dim, device=device)

        # Full Fisher tensor (sparse updates)
        fisher = torch.zeros(num_embeddings, embedding_dim, device=device)

        # Process in blocks
        for start_idx in range(0, len(used_indices), self.block_size):
            end_idx = min(start_idx + self.block_size, len(used_indices))
            block_indices = used_indices[start_idx:end_idx]

            # Get Fisher for this block (from gradients)
            # Note: This requires gradient information
            if embedding_module.weight.grad is not None:
                for idx in block_indices:
                    fisher[idx] = embedding_module.weight.grad[idx] ** 2

        return fisher


def compute_parameter_importance_fast(model, fim_dict, num_domains):
    """
    Fast computation of parameter importance scores

    Categorizes parameters and computes normalized importance
    """
    importance_scores = {}

    # Categorize parameters
    shared_params = []
    specific_params = []

    for name, param in model.named_parameters():
        if 'share' in name or (name.startswith('agg_list') and str(num_domains) in name):
            shared_params.append(name)
        else:
            specific_params.append(name)

    # Compute importance scores
    # Normalize within each category for fair comparison

    # Shared parameters
    shared_total = 0
    shared_count = 0
    for name in shared_params:
        if name in fim_dict:
            shared_total += fim_dict[name].sum().item()
            shared_count += fim_dict[name].numel()

    shared_mean = shared_total / shared_count if shared_count > 0 else 1.0

    # Specific parameters
    specific_total = 0
    specific_count = 0
    for name in specific_params:
        if name in fim_dict:
            specific_total += fim_dict[name].sum().item()
            specific_count += fim_dict[name].numel()

    specific_mean = specific_total / specific_count if specific_count > 0 else 1.0

    # Normalized importance
    for name in fim_dict:
        if name in shared_params:
            importance_scores[name] = fim_dict[name] / (shared_mean + 1e-8)
        else:
            importance_scores[name] = fim_dict[name] / (specific_mean + 1e-8)

    return importance_scores


class EfficientPerturbation:
    """
    Memory-efficient perturbation strategies for DR-FIM computation
    """

    @staticmethod
    def inplace_edge_dropout(batch, dropout_rate=0.1):
        """
        Apply edge dropout in-place (no memory copy)
        """
        user, pos_item, neg_item, context_item, context_score, global_item, global_score = batch

        # In-place masking
        mask = torch.rand_like(context_score) > dropout_rate
        context_score.mul_(mask.float())

        return batch

    @staticmethod
    def fast_noise_injection(batch, noise_scale=0.01):
        """
        Fast noise injection using additive noise
        """
        user, pos_item, neg_item, context_item, context_score, global_item, global_score = batch

        # Add noise in-place
        context_score.add_(torch.randn_like(context_score) * noise_scale)
        context_score.clamp_(min=0)

        return batch

    @staticmethod
    def domain_swap_efficient(batch, domain_id, num_domains):
        """
        Efficient cross-domain context swapping
        """
        user, pos_item, neg_item, context_item, context_score, global_item, global_score = batch

        if num_domains < 2:
            return batch

        # Use view instead of copy for efficiency
        other_domain = (domain_id + 1) % num_domains
        new_context_item = global_item[:, other_domain, :].contiguous()
        new_context_score = global_score[:, other_domain, :].contiguous()

        return (user, pos_item, neg_item, new_context_item, new_context_score,
                global_item, global_score)


def estimate_training_overhead(config, num_params, num_domains):
    """
    Estimate the training time overhead from FisherTune

    Returns:
        overhead_percentage: Estimated % increase in training time
    """
    base_overhead = 0

    if not config.use_fim and not config.use_dr_fim:
        return 0

    # Fisher computation overhead (per update)
    # ~1 extra forward-backward pass per N batches
    fisher_overhead = (config.num_samples_fisher / config.fisher_update_freq) * 2

    # DR-FIM doubles the Fisher computation
    if config.use_dr_fim:
        fisher_overhead *= 2

    # Perturbation overhead (minimal for efficient implementations)
    perturbation_overhead = 0.1  # 10% overhead for perturbation

    # Gradient masking overhead (very small)
    masking_overhead = 0.01 * num_params  # O(num_params) operation

    # Total overhead per epoch
    total_overhead = fisher_overhead + perturbation_overhead + masking_overhead

    # Normalize to percentage
    # Assuming ~500 iterations per epoch, ~2*num_domains forward passes per iteration
    base_ops_per_epoch = 500 * 2 * num_domains
    overhead_percentage = (total_overhead / base_ops_per_epoch) * 100

    # Add scheduling overhead (minimal)
    overhead_percentage += 0.5  # ~0.5% for threshold computation

    return min(overhead_percentage, 15)  # Cap at 15% overhead


def print_efficiency_report(config, model, num_domains):
    """
    Print efficiency report for FisherTune configuration
    """
    num_params = sum(p.numel() for p in model.parameters())

    overhead = estimate_training_overhead(config, num_params, num_domains)

    print("\n" + "=" * 60)
    print("FISHERTUNE EFFICIENCY REPORT")
    print("=" * 60)
    print(f"Total Parameters: {num_params:,}")
    print(f"Number of Domains: {num_domains}")
    print(f"\nConfiguration:")
    print(f"  FIM Enabled: {config.use_fim}")
    print(f"  DR-FIM Enabled: {config.use_dr_fim}")
    print(f"  Scheduling Enabled: {config.use_scheduling}")
    print(f"  Perturbation Type: {config.perturbation_type}")
    print(f"  Fisher Update Frequency: Every {config.fisher_update_freq} epochs")
    print(f"  Samples for Fisher: {config.num_samples_fisher}")

    print(f"\nEstimated Training Overhead: ~{overhead:.1f}%")

    if overhead < 5:
        print("✅ Excellent efficiency - minimal overhead")
    elif overhead < 10:
        print("✅ Good efficiency - acceptable overhead")
    elif overhead < 15:
        print("⚠️  Moderate overhead - consider reducing Fisher samples")
    else:
        print("⚠️  High overhead - recommend optimization")

    print("=" * 60 + "\n")
