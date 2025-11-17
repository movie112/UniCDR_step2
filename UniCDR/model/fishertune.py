"""
FisherTune for UniCDR: Domain-Aware Parameter Selection for Cross-Domain Recommendation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
import copy


class FisherTuneConfig:
    """Configuration for FisherTune experiments"""

    def __init__(self,
                 # Core FisherTune settings
                 use_fim=True,
                 use_dr_fim=False,
                 use_scheduling=True,

                 # Parameter selection strategy
                 param_mode='unified',  # 'unified', 'shared_only', 'specific_only', 'adaptive'

                 # Scheduling parameters
                 delta_min=0.1,
                 delta_max=0.9,
                 schedule_T=10,

                 # For adaptive mode (shared vs specific different thresholds)
                 shared_delta_min=0.7,  # High threshold for shared (mostly frozen)
                 shared_delta_max=0.95,
                 specific_delta_min=0.1,  # Low threshold for specific (more tuning)
                 specific_delta_max=0.5,

                 # Variational inference parameters
                 use_variational=True,
                 prior_variance=1.0,
                 gamma=1.0,

                 # Perturbation strategy
                 perturbation_type='cross_domain',  # 'none', 'edge_dropout', 'popularity_weight', 'noise', 'cross_domain'
                 perturbation_rate=0.1,
                 noise_scale=0.01,

                 # When to start FisherTune (after initial learning)
                 warmup_epochs=5,
                 fisher_update_freq=5,  # How often to recompute Fisher

                 # Efficiency settings
                 num_samples_fisher=100,  # Number of samples for Fisher estimation
                 diagonal_fisher=True,

                 # Device
                 device='cuda'):

        self.use_fim = use_fim
        self.use_dr_fim = use_dr_fim
        self.use_scheduling = use_scheduling
        self.param_mode = param_mode

        self.delta_min = delta_min
        self.delta_max = delta_max
        self.schedule_T = schedule_T

        self.shared_delta_min = shared_delta_min
        self.shared_delta_max = shared_delta_max
        self.specific_delta_min = specific_delta_min
        self.specific_delta_max = specific_delta_max

        self.use_variational = use_variational
        self.prior_variance = prior_variance
        self.gamma = gamma

        self.perturbation_type = perturbation_type
        self.perturbation_rate = perturbation_rate
        self.noise_scale = noise_scale

        self.warmup_epochs = warmup_epochs
        self.fisher_update_freq = fisher_update_freq

        self.num_samples_fisher = num_samples_fisher
        self.diagonal_fisher = diagonal_fisher

        self.device = device


class FisherTuneModule:
    """
    FisherTune implementation for UniCDR
    Computes Fisher Information Matrix and applies domain-aware parameter selection
    """

    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = config.device

        # Store Fisher information for each parameter
        self.fim = {}
        self.dr_fim = {}
        self.variational_precision = {}  # Lambda in the paper

        # Parameter masks for selective updating
        self.param_masks = {}

        # Track which parameters are shared vs domain-specific
        self._categorize_parameters()

        # Initialize Fisher information
        self._initialize_fisher()

        # Current epoch for scheduling
        self.current_epoch = 0

    def _categorize_parameters(self):
        """Categorize parameters as shared or domain-specific"""
        self.shared_params = []
        self.specific_params = []

        for name, param in self.model.named_parameters():
            if 'share' in name or (name.startswith('agg_list') and
                                   str(self.model.opt["num_domains"]) in name):
                # Shared parameters: share_user_embedding, share_item_embedding, shared aggregator
                self.shared_params.append(name)
            else:
                # Domain-specific: specific embeddings, per-domain aggregators, discriminators
                self.specific_params.append(name)

        print(f"FisherTune: Found {len(self.shared_params)} shared params, "
              f"{len(self.specific_params)} domain-specific params")

    def _initialize_fisher(self):
        """Initialize Fisher information matrices"""
        for name, param in self.model.named_parameters():
            if self.config.diagonal_fisher:
                self.fim[name] = torch.zeros_like(param.data)
                self.dr_fim[name] = torch.zeros_like(param.data)
                self.variational_precision[name] = torch.ones_like(param.data)
            else:
                # Full Fisher (not recommended for large models)
                self.fim[name] = torch.zeros(param.numel(), param.numel(), device=self.device)

            # Initialize masks to all ones (update all)
            self.param_masks[name] = torch.ones_like(param.data, dtype=torch.bool)

    def compute_fisher_information(self, dataloader_dict, criterion, num_samples=None):
        """
        Compute diagonal Fisher Information Matrix

        FIM_i = E[(d L/d theta_i)^2]
        """
        if num_samples is None:
            num_samples = self.config.num_samples_fisher

        # Reset Fisher
        for name in self.fim:
            self.fim[name].zero_()

        self.model.eval()
        sample_count = 0

        # Sample from all domains
        for domain_id, dataloader in dataloader_dict.items():
            if sample_count >= num_samples:
                break

            for batch in dataloader:
                if sample_count >= num_samples:
                    break

                self.model.zero_grad()

                # Compute loss
                loss = self._compute_batch_loss(domain_id, batch, criterion)

                # Compute gradients
                loss.backward()

                # Accumulate squared gradients (diagonal Fisher)
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        self.fim[name] += param.grad.data ** 2

                sample_count += 1

        # Average
        for name in self.fim:
            self.fim[name] /= sample_count

        print(f"FisherTune: Computed Fisher Information from {sample_count} samples")

    def compute_dr_fim(self, dataloader_dict, perturbed_dataloader_dict, criterion, num_samples=None):
        """
        Compute Domain-Related Fisher Information Matrix (DR-FIM)

        DR-FIM = FIM(x,y) + exp(-eps) * |FIM(x,y) - FIM(x',y)| / min(FIM(x), FIM(x')) + eps

        where x' is the perturbed sample simulating domain shift
        """
        if num_samples is None:
            num_samples = self.config.num_samples_fisher

        # First compute standard FIM on original data
        self.compute_fisher_information(dataloader_dict, criterion, num_samples)

        # Store original FIM
        original_fim = {}
        for name in self.fim:
            original_fim[name] = self.fim[name].clone()

        # Compute FIM on perturbed data
        for name in self.fim:
            self.fim[name].zero_()

        self.model.eval()
        sample_count = 0

        for domain_id, dataloader in perturbed_dataloader_dict.items():
            if sample_count >= num_samples:
                break

            for batch in dataloader:
                if sample_count >= num_samples:
                    break

                self.model.zero_grad()
                loss = self._compute_batch_loss(domain_id, batch, criterion)
                loss.backward()

                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        self.fim[name] += param.grad.data ** 2

                sample_count += 1

        for name in self.fim:
            self.fim[name] /= sample_count

        # Compute DR-FIM
        epsilon = 1e-8
        perturbation_strength = self.config.perturbation_rate

        for name in self.dr_fim:
            # Delta F = |F(x) - F(x')| / min(F(x), F(x'))
            min_fim = torch.min(original_fim[name], self.fim[name]) + epsilon
            delta_f = torch.abs(original_fim[name] - self.fim[name]) / min_fim

            # DR-FIM = F(x) + exp(-perturbation) * Delta F
            exp_factor = np.exp(-perturbation_strength)
            self.dr_fim[name] = original_fim[name] + exp_factor * delta_f

        # Restore original FIM
        for name in self.fim:
            self.fim[name] = original_fim[name]

        print(f"FisherTune: Computed DR-FIM from {sample_count} perturbed samples")

    def compute_variational_fisher(self, dataloader_dict, criterion, num_samples=None):
        """
        Compute variational Fisher using posterior precision (Lambda)

        This provides more stable estimates by incorporating prior information

        F_theta = gamma * (Lambda - tau^-2 * I)
        """
        if num_samples is None:
            num_samples = self.config.num_samples_fisher

        # First compute standard FIM
        self.compute_fisher_information(dataloader_dict, criterion, num_samples)

        # Apply variational regularization
        tau_sq = self.config.prior_variance
        gamma = self.config.gamma

        for name in self.variational_precision:
            # Lambda = FIM + prior precision
            prior_precision = 1.0 / tau_sq
            self.variational_precision[name] = self.fim[name] + prior_precision

            # Stable Fisher estimate
            # F = gamma * (Lambda - tau^-2 * I)
            self.fim[name] = gamma * (self.variational_precision[name] - prior_precision)
            self.fim[name] = torch.clamp(self.fim[name], min=0)  # Ensure non-negative

        print("FisherTune: Applied variational stabilization to Fisher estimates")

    def _compute_batch_loss(self, domain_id, batch, criterion):
        """Compute loss for a single batch"""
        if self.model.opt["cuda"]:
            batch = [b.cuda() for b in batch]

        user, pos_item, neg_item, context_item, context_score, global_item, global_score = batch

        # Forward pass
        self.model.item_embedding_select()
        user_feature = self.model.forward_user(domain_id, user, context_item, context_score,
                                               global_item, global_score)
        pos_item_feature = self.model.forward_item(domain_id, pos_item)
        neg_item_feature = self.model.forward_item(domain_id, neg_item)

        pos_score = self.model.predict_dot(user_feature, pos_item_feature)
        neg_score = self.model.predict_dot(user_feature, neg_item_feature)

        pos_labels = torch.ones_like(pos_score)
        neg_labels = torch.zeros_like(neg_score)

        loss = criterion(pos_score, pos_labels) + criterion(neg_score, neg_labels)

        return loss

    def update_parameter_masks(self, epoch):
        """
        Update parameter masks based on Fisher information and scheduling

        Parameters with high Fisher values are updated; others are frozen
        """
        self.current_epoch = epoch

        if not self.config.use_scheduling:
            # No scheduling - update all based on current Fisher threshold
            threshold = self.config.delta_min
            self._apply_threshold(threshold)
            return

        # Compute time-dependent threshold
        t = epoch - self.config.warmup_epochs
        if t < 0:
            # During warmup, update all parameters
            for name in self.param_masks:
                self.param_masks[name].fill_(True)
            return

        if self.config.param_mode == 'adaptive':
            # Different thresholds for shared vs specific
            self._apply_adaptive_threshold(t)
        else:
            # Unified threshold with exponential decay
            delta_t = self.config.delta_min + (self.config.delta_max - self.config.delta_min) * \
                      np.exp(-t / self.config.schedule_T)
            self._apply_threshold(delta_t)

    def _apply_threshold(self, threshold):
        """Apply Fisher threshold to select parameters"""
        # Use DR-FIM if available, otherwise standard FIM
        fim_to_use = self.dr_fim if self.config.use_dr_fim else self.fim

        total_params = 0
        selected_params = 0

        for name in self.param_masks:
            # Normalize Fisher values for this parameter
            fim_values = fim_to_use[name]
            if fim_values.max() > 0:
                normalized_fim = fim_values / (fim_values.max() + 1e-8)
            else:
                normalized_fim = fim_values

            # Select parameters above threshold
            if self.config.param_mode == 'shared_only':
                if name in self.shared_params:
                    self.param_masks[name] = normalized_fim > threshold
                else:
                    self.param_masks[name].fill_(False)
            elif self.config.param_mode == 'specific_only':
                if name in self.specific_params:
                    self.param_masks[name] = normalized_fim > threshold
                else:
                    self.param_masks[name].fill_(False)
            else:  # unified
                self.param_masks[name] = normalized_fim > threshold

            total_params += self.param_masks[name].numel()
            selected_params += self.param_masks[name].sum().item()

        selection_ratio = selected_params / total_params if total_params > 0 else 0
        print(f"FisherTune: Selected {selected_params}/{total_params} parameters "
              f"({selection_ratio:.2%}) with threshold {threshold:.4f}")

    def _apply_adaptive_threshold(self, t):
        """Apply different thresholds for shared vs specific parameters"""
        fim_to_use = self.dr_fim if self.config.use_dr_fim else self.fim

        # Shared: high threshold (conservative, mostly frozen)
        shared_delta_t = self.config.shared_delta_min + \
                        (self.config.shared_delta_max - self.config.shared_delta_min) * \
                        np.exp(-t / self.config.schedule_T)

        # Specific: low threshold (aggressive, more tuning)
        specific_delta_t = self.config.specific_delta_min + \
                          (self.config.specific_delta_max - self.config.specific_delta_min) * \
                          np.exp(-t / self.config.schedule_T)

        shared_selected = 0
        shared_total = 0
        specific_selected = 0
        specific_total = 0

        for name in self.param_masks:
            fim_values = fim_to_use[name]
            if fim_values.max() > 0:
                normalized_fim = fim_values / (fim_values.max() + 1e-8)
            else:
                normalized_fim = fim_values

            if name in self.shared_params:
                self.param_masks[name] = normalized_fim > shared_delta_t
                shared_total += self.param_masks[name].numel()
                shared_selected += self.param_masks[name].sum().item()
            else:
                self.param_masks[name] = normalized_fim > specific_delta_t
                specific_total += self.param_masks[name].numel()
                specific_selected += self.param_masks[name].sum().item()

        shared_ratio = shared_selected / shared_total if shared_total > 0 else 0
        specific_ratio = specific_selected / specific_total if specific_total > 0 else 0

        print(f"FisherTune Adaptive: Shared {shared_selected}/{shared_total} ({shared_ratio:.2%}) "
              f"thresh={shared_delta_t:.4f}; "
              f"Specific {specific_selected}/{specific_total} ({specific_ratio:.2%}) "
              f"thresh={specific_delta_t:.4f}")

    def apply_gradient_mask(self):
        """Apply masks to gradients after backward pass"""
        for name, param in self.model.named_parameters():
            if param.grad is not None and name in self.param_masks:
                # Zero out gradients for non-selected parameters
                param.grad.data *= self.param_masks[name].float()

    def get_statistics(self):
        """Get statistics about Fisher information and parameter selection"""
        stats = {}

        # Compute average Fisher for shared vs specific
        shared_fim_sum = 0
        shared_count = 0
        specific_fim_sum = 0
        specific_count = 0

        fim_to_use = self.dr_fim if self.config.use_dr_fim else self.fim

        for name in fim_to_use:
            if name in self.shared_params:
                shared_fim_sum += fim_to_use[name].sum().item()
                shared_count += fim_to_use[name].numel()
            else:
                specific_fim_sum += fim_to_use[name].sum().item()
                specific_count += fim_to_use[name].numel()

        stats['avg_shared_fim'] = shared_fim_sum / shared_count if shared_count > 0 else 0
        stats['avg_specific_fim'] = specific_fim_sum / specific_count if specific_count > 0 else 0

        # Selection statistics
        shared_selected = 0
        specific_selected = 0

        for name in self.param_masks:
            if name in self.shared_params:
                shared_selected += self.param_masks[name].sum().item()
            else:
                specific_selected += self.param_masks[name].sum().item()

        stats['shared_selection_ratio'] = shared_selected / shared_count if shared_count > 0 else 0
        stats['specific_selection_ratio'] = specific_selected / specific_count if specific_count > 0 else 0

        return stats


class DomainPerturbation:
    """
    Perturbation strategies for simulating domain shift in UniCDR
    """

    def __init__(self, config, opt):
        self.config = config
        self.opt = opt
        self.device = config.device

    def perturb_batch(self, batch, domain_id, perturbation_type=None):
        """
        Apply perturbation to a batch to simulate domain shift
        """
        if perturbation_type is None:
            perturbation_type = self.config.perturbation_type

        if perturbation_type == 'none':
            return batch
        elif perturbation_type == 'edge_dropout':
            return self._edge_dropout(batch)
        elif perturbation_type == 'popularity_weight':
            return self._popularity_weight(batch)
        elif perturbation_type == 'noise':
            return self._add_noise(batch)
        elif perturbation_type == 'cross_domain':
            return self._cross_domain_swap(batch, domain_id)
        else:
            return batch

    def _edge_dropout(self, batch):
        """
        Drop edges (interactions) randomly to simulate sparse domain
        """
        user, pos_item, neg_item, context_item, context_score, global_item, global_score = batch

        # Randomly mask out some context items
        dropout_rate = self.config.perturbation_rate
        mask = torch.rand_like(context_score) > dropout_rate

        perturbed_context_score = context_score * mask.float()

        return (user, pos_item, neg_item, context_item, perturbed_context_score,
                global_item, global_score)

    def _popularity_weight(self, batch):
        """
        Reweight edges based on inverse popularity (simulate long-tail shift)
        """
        user, pos_item, neg_item, context_item, context_score, global_item, global_score = batch

        # Apply inverse frequency weighting to context scores
        # This simulates a domain where popular items are less emphasized
        perturbation_factor = 1.0 - self.config.perturbation_rate

        # Inverse the score distribution (high scores become low)
        max_score = context_score.max() + 1e-8
        inverted_score = max_score - context_score
        perturbed_context_score = perturbation_factor * context_score + \
                                  (1 - perturbation_factor) * inverted_score

        return (user, pos_item, neg_item, context_item, perturbed_context_score,
                global_item, global_score)

    def _add_noise(self, batch):
        """
        Add Gaussian noise to scores to simulate noisy domain
        """
        user, pos_item, neg_item, context_item, context_score, global_item, global_score = batch

        # Add noise to context scores
        noise = torch.randn_like(context_score) * self.config.noise_scale
        perturbed_context_score = context_score + noise
        perturbed_context_score = torch.clamp(perturbed_context_score, min=0)

        # Add noise to global scores
        global_noise = torch.randn_like(global_score) * self.config.noise_scale
        perturbed_global_score = global_score + global_noise
        perturbed_global_score = torch.clamp(perturbed_global_score, min=0)

        return (user, pos_item, neg_item, context_item, perturbed_context_score,
                global_item, perturbed_global_score)

    def _cross_domain_swap(self, batch, domain_id):
        """
        Swap context with other domain's context to simulate actual domain shift
        """
        user, pos_item, neg_item, context_item, context_score, global_item, global_score = batch

        num_domains = self.opt["num_domains"]
        if num_domains < 2:
            return batch

        # Swap with another domain's global context
        other_domain = (domain_id + 1) % num_domains

        # Use other domain's items as context (simulating domain shift)
        # This is a stronger perturbation that uses actual cross-domain information
        perturbed_context_item = global_item[:, other_domain, :]
        perturbed_context_score = global_score[:, other_domain, :]

        return (user, pos_item, neg_item, perturbed_context_item, perturbed_context_score,
                global_item, global_score)

    def create_perturbed_dataloader(self, original_dataloader_dict):
        """
        Create a perturbed version of the dataloaders
        """
        perturbed_dict = {}

        for domain_id, dataloader in original_dataloader_dict.items():
            perturbed_dict[domain_id] = PerturbedDataLoader(
                dataloader, self, domain_id
            )

        return perturbed_dict


class PerturbedDataLoader:
    """Wrapper around dataloader that applies perturbation"""

    def __init__(self, original_dataloader, perturbation_module, domain_id):
        self.original_dataloader = original_dataloader
        self.perturbation = perturbation_module
        self.domain_id = domain_id

    def __iter__(self):
        for batch in self.original_dataloader:
            yield self.perturbation.perturb_batch(batch, self.domain_id)

    def __len__(self):
        return len(self.original_dataloader)


def get_experiment_config(experiment_type):
    """
    Factory function to create configurations for different experiments

    experiment_type:
        - 'fim_only': Standard FIM with scheduling
        - 'dr_fim': Domain-Related FIM
        - 'unified': All parameters unified (no shared/specific distinction)
        - 'shared_only': Only tune shared parameters
        - 'specific_only': Only tune domain-specific parameters
        - 'adaptive': Shared frozen (high delta), specific tuned (low delta)
        - 'perturbation_edge_dropout': DR-FIM with edge dropout
        - 'perturbation_popularity': DR-FIM with popularity weighting
        - 'perturbation_noise': DR-FIM with noise injection
        - 'perturbation_cross_domain': DR-FIM with actual domain pairs
    """

    if experiment_type == 'fim_only':
        return FisherTuneConfig(
            use_fim=True,
            use_dr_fim=False,
            use_scheduling=True,
            param_mode='unified',
            delta_min=0.1,
            delta_max=0.9,
            perturbation_type='none'
        )

    elif experiment_type == 'dr_fim':
        return FisherTuneConfig(
            use_fim=True,
            use_dr_fim=True,
            use_scheduling=True,
            param_mode='unified',
            delta_min=0.1,
            delta_max=0.9,
            perturbation_type='noise'
        )

    elif experiment_type == 'unified':
        return FisherTuneConfig(
            use_fim=True,
            use_dr_fim=True,
            use_scheduling=True,
            param_mode='unified',
            delta_min=0.1,
            delta_max=0.9,
            perturbation_type='cross_domain'
        )

    elif experiment_type == 'shared_only':
        return FisherTuneConfig(
            use_fim=True,
            use_dr_fim=True,
            use_scheduling=True,
            param_mode='shared_only',
            delta_min=0.1,
            delta_max=0.9,
            perturbation_type='cross_domain'
        )

    elif experiment_type == 'specific_only':
        return FisherTuneConfig(
            use_fim=True,
            use_dr_fim=True,
            use_scheduling=True,
            param_mode='specific_only',
            delta_min=0.1,
            delta_max=0.9,
            perturbation_type='cross_domain'
        )

    elif experiment_type == 'adaptive':
        return FisherTuneConfig(
            use_fim=True,
            use_dr_fim=True,
            use_scheduling=True,
            param_mode='adaptive',
            # Shared: high threshold = mostly frozen
            shared_delta_min=0.7,
            shared_delta_max=0.95,
            # Specific: low threshold = more tuning
            specific_delta_min=0.1,
            specific_delta_max=0.5,
            perturbation_type='cross_domain'
        )

    elif experiment_type == 'perturbation_edge_dropout':
        return FisherTuneConfig(
            use_fim=True,
            use_dr_fim=True,
            use_scheduling=True,
            param_mode='unified',
            perturbation_type='edge_dropout',
            perturbation_rate=0.2
        )

    elif experiment_type == 'perturbation_popularity':
        return FisherTuneConfig(
            use_fim=True,
            use_dr_fim=True,
            use_scheduling=True,
            param_mode='unified',
            perturbation_type='popularity_weight',
            perturbation_rate=0.3
        )

    elif experiment_type == 'perturbation_noise':
        return FisherTuneConfig(
            use_fim=True,
            use_dr_fim=True,
            use_scheduling=True,
            param_mode='unified',
            perturbation_type='noise',
            noise_scale=0.05
        )

    elif experiment_type == 'perturbation_cross_domain':
        return FisherTuneConfig(
            use_fim=True,
            use_dr_fim=True,
            use_scheduling=True,
            param_mode='unified',
            perturbation_type='cross_domain'
        )

    elif experiment_type == 'baseline':
        # No FisherTune - standard UniCDR training
        return FisherTuneConfig(
            use_fim=False,
            use_dr_fim=False,
            use_scheduling=False,
            param_mode='unified'
        )

    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")
