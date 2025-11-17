"""
FisherTune Trainer for UniCDR.

Integrates all FisherTune components:
- Domain-Related Fisher Information computation
- Parameter selection based on DR-FIM
- Variational inference for stable estimation
- Domain perturbation strategies
- Shared/Specific parameter differentiation
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from typing import Dict, List, Optional, Tuple

from model.UniCDR import UniCDR
from model.trainer import CrossTrainer
from utils import torch_utils

from .fisher_info import (
    FisherInformationComputer,
    DomainRelatedFIM,
    MultiDomainFIM,
    separate_fim_by_type
)
from .domain_perturbation import (
    DomainPerturbation,
    EmbeddingPerturbation,
    PopularityComputer
)
from .variational_fisher import (
    VariationalFisher,
    AdaptiveVariationalFisher,
    OnlineFisherEstimator
)
from .parameter_scheduler import (
    SharedSpecificScheduler,
    AdaptiveScheduler
)
from .fishertune_optimizer import (
    FisherTuneOptimizer,
    FisherRegularizedOptimizer
)


class FisherTuneTrainer(CrossTrainer):
    """
    FisherTune-enhanced trainer for UniCDR.

    Key features:
    1. Progressive parameter selection based on DR-FIM
    2. Separate thresholds for shared vs specific parameters
    3. Domain perturbation for robust FIM estimation
    4. Variational inference for stability
    """

    def __init__(self, opt):
        """
        Initialize FisherTune trainer.

        Additional opt keys:
        - use_fishertune: Enable FisherTune (default: True)
        - fishertune_warmup: Warmup epochs before FisherTune (default: 10)
        - shared_delta_min/max: Threshold range for shared params
        - specific_delta_min/max: Threshold range for specific params
        - fim_update_freq: How often to update FIM (default: 10)
        - perturbation_type: Type of domain perturbation
        - variational_tau/gamma: VI hyperparameters
        - shared_lr/specific_lr: Learning rates
        - use_ewc: Use EWC regularization
        """
        # Initialize base model
        self.opt = opt
        if self.opt["model"] == "UniCDR":
            self.model = UniCDR(opt)
        else:
            print("please input right model name!")
            exit(0)

        self.criterion = nn.BCEWithLogitsLoss()
        if opt['cuda']:
            self.model.cuda()
            self.criterion.cuda()

        self.device = 'cuda' if opt['cuda'] else 'cpu'

        # FisherTune configuration
        self.use_fishertune = opt.get('use_fishertune', True)
        self.fishertune_warmup = opt.get('fishertune_warmup', 10)
        self.fim_update_freq = opt.get('fim_update_freq', 10)
        self.num_domains = opt.get('num_domain', 2)

        # Initialize FisherTune components
        if self.use_fishertune:
            self._init_fishertune_components()
        else:
            # Fallback to standard optimizer
            self.optimizer = torch_utils.get_optimizer(
                opt['optim'], self.model.parameters(),
                opt['lr'], opt["weight_decay"]
            )

        self.epoch_rec_loss = []
        self.current_epoch = 0
        self.training_step = 0

    def _init_fishertune_components(self):
        """Initialize all FisherTune components."""
        opt = self.opt

        # 1. Fisher Information Computer
        self.fim_computer = FisherInformationComputer(
            self.model, self.device
        )
        self.dr_fim_computer = DomainRelatedFIM(epsilon=1e-8)
        self.multi_domain_fim = MultiDomainFIM(
            self.model, self.num_domains, self.device
        )

        # 2. Domain Perturbation
        self.perturbation = DomainPerturbation(
            perturbation_type=opt.get('perturbation_type', 'combined'),
            edge_dropout_rate=opt.get('edge_dropout_rate', 0.2),
            noise_std=opt.get('perturbation_noise_std', 0.1),
            popularity_alpha=opt.get('popularity_alpha', 0.5),
            device=self.device
        )
        self.embedding_perturbation = EmbeddingPerturbation(
            noise_std=opt.get('embedding_noise_std', 0.1)
        )

        # 3. Variational Fisher (for stability)
        use_adaptive_vi = opt.get('use_adaptive_vi', True)
        if use_adaptive_vi:
            self.variational_fisher = AdaptiveVariationalFisher(
                self.model,
                tau_init=opt.get('variational_tau', 1.0),
                gamma_init=opt.get('variational_gamma', 0.1),
                tau_decay=opt.get('tau_decay', 0.99),
                gamma_growth=opt.get('gamma_growth', 1.01),
                device=self.device
            )
        else:
            self.variational_fisher = VariationalFisher(
                self.model,
                tau=opt.get('variational_tau', 1.0),
                gamma=opt.get('variational_gamma', 0.1),
                device=self.device
            )

        # Online Fisher estimator
        self.online_fisher = OnlineFisherEstimator(
            self.model,
            momentum=opt.get('fisher_momentum', 0.9),
            damping=opt.get('fisher_damping', 1e-4),
            device=self.device
        )

        # 4. Parameter Scheduler
        use_adaptive_scheduler = opt.get('use_adaptive_scheduler', False)
        if use_adaptive_scheduler:
            self.param_scheduler = AdaptiveScheduler(
                self.model,
                shared_delta_min=opt.get('shared_delta_min', 0.5),
                shared_delta_max=opt.get('shared_delta_max', 0.95),
                specific_delta_min=opt.get('specific_delta_min', 0.1),
                specific_delta_max=opt.get('specific_delta_max', 0.7),
                decay_constant=opt.get('threshold_decay_constant', 50.0),
                warmup_epochs=self.fishertune_warmup,
                adaptation_rate=opt.get('scheduler_adaptation_rate', 0.1),
                device=self.device
            )
        else:
            self.param_scheduler = SharedSpecificScheduler(
                self.model,
                shared_delta_min=opt.get('shared_delta_min', 0.5),
                shared_delta_max=opt.get('shared_delta_max', 0.95),
                specific_delta_min=opt.get('specific_delta_min', 0.1),
                specific_delta_max=opt.get('specific_delta_max', 0.7),
                decay_constant=opt.get('threshold_decay_constant', 50.0),
                warmup_epochs=self.fishertune_warmup,
                device=self.device
            )

        # 5. Optimizer
        use_fisher_reg = opt.get('use_fisher_regularization', False)
        if use_fisher_reg:
            self.ft_optimizer = FisherRegularizedOptimizer(
                self.model,
                base_optimizer_class=self._get_optimizer_class(opt['optim']),
                shared_lr=opt.get('shared_lr', 0.0001),
                specific_lr=opt.get('specific_lr', 0.001),
                shared_weight_decay=opt.get('shared_weight_decay', 1e-3),
                specific_weight_decay=opt.get('specific_weight_decay', 1e-5),
                fisher_reg_strength=opt.get('fisher_reg_strength', 0.1)
            )
        else:
            self.ft_optimizer = FisherTuneOptimizer(
                self.model,
                base_optimizer_class=self._get_optimizer_class(opt['optim']),
                shared_lr=opt.get('shared_lr', 0.0001),
                specific_lr=opt.get('specific_lr', 0.001),
                shared_weight_decay=opt.get('shared_weight_decay', 1e-3),
                specific_weight_decay=opt.get('specific_weight_decay', 1e-5),
                use_natural_gradient=opt.get('use_natural_gradient', False),
                fisher_damping=opt.get('fisher_damping', 1e-4)
            )

        # Make optimizer accessible via standard name
        self.optimizer = self.ft_optimizer.optimizer

        # Store DR-FIM
        self.current_dr_fim = {}

        # Domain popularity (computed later)
        self.domain_popularity = {}

        # Logging
        self.fim_history = []
        self.selection_history = []

    def _get_optimizer_class(self, optim_name: str):
        """Get optimizer class from name."""
        if optim_name.lower() == 'sgd':
            return torch.optim.SGD
        elif optim_name.lower() == 'adam':
            return torch.optim.Adam
        elif optim_name.lower() == 'adamax':
            return torch.optim.Adamax
        elif optim_name.lower() == 'adagrad':
            return torch.optim.Adagrad
        else:
            return torch.optim.Adam

    def set_domain_data(
        self,
        domain_loaders: List,
        domain_interactions: Optional[Dict] = None,
        item_max: Optional[List[int]] = None
    ):
        """
        Set domain data for FIM computation.

        Args:
            domain_loaders: Data loaders for each domain
            domain_interactions: Interaction data for popularity computation
            item_max: Maximum item ID per domain
        """
        self.domain_loaders = domain_loaders

        # Compute item popularity if available
        if domain_interactions and item_max:
            for domain_id in range(self.num_domains):
                if domain_id < len(item_max):
                    self.domain_popularity[domain_id] = \
                        PopularityComputer.compute_from_interactions(
                            domain_interactions.get(domain_id, {}),
                            item_max[domain_id]
                        ).to(self.device)

    def reconstruct_graph(self, domain_id, batch):
        """
        Compute loss for a batch with optional FisherTune enhancements.
        """
        user, pos_item, neg_item, context_item, context_score, \
            global_item, global_score = self.unpack_batch(batch)

        user_feature = self.model.forward_user(
            domain_id, user, context_item, context_score,
            global_item, global_score
        )
        pos_item_feature = self.model.forward_item(domain_id, pos_item)
        neg_item_feature = self.model.forward_item(domain_id, neg_item)

        pos_score = self.model.predict_dot(user_feature, pos_item_feature)
        neg_score = self.model.predict_dot(user_feature, neg_item_feature)

        pos_labels = torch.ones(pos_score.size())
        neg_labels = torch.zeros(neg_score.size())

        if self.opt["cuda"]:
            pos_labels = pos_labels.cuda()
            neg_labels = neg_labels.cuda()

        # Base loss
        base_loss = self.opt["lambda_loss"] * (
            self.criterion(pos_score, pos_labels) +
            self.criterion(neg_score, neg_labels)
        ) + (1 - self.opt["lambda_loss"]) * self.model.critic_loss

        # Add Fisher regularization if enabled
        if self.use_fishertune and isinstance(self.ft_optimizer, FisherRegularizedOptimizer):
            base_loss = self.ft_optimizer.get_regularized_loss(base_loss)

        # Add variational KL if in training mode
        if self.use_fishertune and self.training_step > 0:
            if self.opt.get('use_variational_loss', False):
                base_loss = self.variational_fisher.compute_variational_loss(base_loss)

        return base_loss

    def train_step(self, domain_id: int, batch):
        """
        Perform single training step with FisherTune.

        Args:
            domain_id: Current domain ID
            batch: Training batch

        Returns:
            Loss value
        """
        self.model.train()
        self.ft_optimizer.zero_grad()

        loss = self.reconstruct_graph(domain_id, batch)

        loss.backward()

        # Update online Fisher estimate
        if self.use_fishertune and self.current_epoch >= self.fishertune_warmup:
            with torch.no_grad():
                self.online_fisher.update(loss.detach())

        # Apply FisherTune parameter selection
        if self.use_fishertune and self.current_epoch >= self.fishertune_warmup:
            self._apply_parameter_selection()

        self.ft_optimizer.step()

        self.training_step += 1

        return loss.item()

    def _apply_parameter_selection(self):
        """Apply FisherTune parameter selection before optimizer step."""
        if not self.current_dr_fim:
            # Use online Fisher as fallback
            self.current_dr_fim = self.online_fisher.get_fisher()

        # Update scheduler with current DR-FIM
        self.param_scheduler.update_dr_fim(self.current_dr_fim)

        # Select parameters
        active_params = self.param_scheduler.select_parameters(
            self.current_epoch,
            percentile_based=self.opt.get('percentile_based_selection', True)
        )

        # Update optimizer
        self.ft_optimizer.update_active_params(active_params)

        # Update Fisher info for natural gradient
        if self.opt.get('use_natural_gradient', False):
            self.ft_optimizer.update_fisher_info(self.current_dr_fim)

    def update_fim(self, num_samples: int = 50):
        """
        Update DR-FIM estimates.
        Should be called periodically (e.g., every few epochs).

        Args:
            num_samples: Number of samples per domain
        """
        if not self.use_fishertune or not hasattr(self, 'domain_loaders'):
            return

        print("Updating Fisher Information...")

        # Method 1: Compute FIM per domain and cross-domain DR-FIM
        if self.opt.get('fim_method', 'online') == 'batch':
            self._update_fim_batch(num_samples)
        else:
            # Use online estimates (already updated during training)
            self.current_dr_fim = self.online_fisher.get_fisher()

        # Update variational precision
        if hasattr(self, 'variational_fisher'):
            for name, fim_val in self.current_dr_fim.items():
                if name in self.variational_fisher.lambda_precision:
                    # Blend with variational estimate
                    blend_factor = 0.5
                    self.variational_fisher.lambda_precision[name] = (
                        blend_factor * fim_val +
                        (1 - blend_factor) * self.variational_fisher.lambda_precision[name]
                    )

        # Log
        self.fim_history.append({
            'epoch': self.current_epoch,
            'mean_fim': np.mean([v.mean().item() for v in self.current_dr_fim.values()]),
            'num_params': len(self.current_dr_fim)
        })

        print(f"FIM updated: {len(self.current_dr_fim)} parameter groups")

    def _update_fim_batch(self, num_samples: int):
        """Update FIM using batch computation across domains."""
        # Compute per-domain FIM
        domain_fims = {}

        for domain_id in range(self.num_domains):
            if domain_id >= len(self.domain_loaders):
                continue

            self.fim_computer.reset_accumulation()
            loader = self.domain_loaders[domain_id]
            sample_count = 0

            for batch in loader:
                if sample_count >= num_samples:
                    break

                loss = self.reconstruct_graph(domain_id, batch)
                self.fim_computer.accumulate_fim(loss, retain_graph=False)
                sample_count += 1

            domain_fims[domain_id] = self.fim_computer.get_accumulated_fim()

        # Compute cross-domain DR-FIM
        if len(domain_fims) >= 2:
            self.multi_domain_fim.domain_fims = domain_fims
            self.current_dr_fim = self.multi_domain_fim.compute_aggregated_dr_fim(
                perturbation_noise=self.opt.get('perturbation_noise', 0.1)
            )
        elif domain_fims:
            # Single domain case
            self.current_dr_fim = list(domain_fims.values())[0]

    def on_epoch_start(self, epoch: int):
        """
        Called at the start of each epoch.

        Args:
            epoch: Current epoch number
        """
        self.current_epoch = epoch

        if self.use_fishertune and epoch >= self.fishertune_warmup:
            # Update FIM periodically
            if epoch % self.fim_update_freq == 0:
                self.update_fim()

            # Adapt VI hyperparameters
            if hasattr(self.variational_fisher, 'adapt_hyperparameters'):
                self.variational_fisher.adapt_hyperparameters()

            # Log selection stats
            stats = self.param_scheduler.get_selection_stats()
            if stats:
                print(f"[FisherTune] Epoch {epoch}: "
                      f"Threshold={stats.get('current_threshold', 0):.4f}, "
                      f"Active params={stats.get('num_active_params', 0)}, "
                      f"Ratio={stats.get('selection_ratio', 0):.2%}")

    def on_epoch_end(self, epoch: int, val_metrics: Optional[Dict] = None):
        """
        Called at the end of each epoch.

        Args:
            epoch: Current epoch
            val_metrics: Validation metrics (if available)
        """
        # Adapt scheduler if using adaptive version
        if self.use_fishertune and hasattr(self.param_scheduler, 'adapt_thresholds'):
            if val_metrics:
                # Compute average gradient norm
                grad_norm = 0.0
                for p in self.model.parameters():
                    if p.grad is not None:
                        grad_norm += p.grad.norm().item()

                avg_loss = np.mean(self.epoch_rec_loss) if self.epoch_rec_loss else 0
                self.param_scheduler.adapt_thresholds(avg_loss, grad_norm)

    def update_lr(self, new_lr):
        """Update learning rate (override for FisherTune)."""
        if self.use_fishertune:
            # Decay both shared and specific LRs proportionally
            decay_factor = new_lr / self.opt['lr']
            self.ft_optimizer.set_lr(
                shared_lr=self.ft_optimizer.shared_lr * decay_factor,
                specific_lr=self.ft_optimizer.specific_lr * decay_factor
            )
        else:
            torch_utils.change_lr(self.optimizer, new_lr)

    def save(self, filename, epoch=None):
        """Save model with FisherTune state."""
        params = {
            'model': self.model.state_dict(),
            'config': self.opt,
        }

        if self.use_fishertune:
            # Save FisherTune specific state
            params['fishertune'] = {
                'dr_fim': {k: v.cpu() for k, v in self.current_dr_fim.items()},
                'fim_history': self.fim_history,
                'selection_history': self.param_scheduler.selection_history,
                'epoch': self.current_epoch
            }

        try:
            torch.save(params, filename)
            print(f"model saved to {filename}")
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")

    def load(self, filename):
        """Load model with FisherTune state."""
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print(f"Cannot load model from {filename}")
            exit()

        self.model.load_state_dict(checkpoint['model'])
        self.opt = checkpoint['config']

        if self.use_fishertune and 'fishertune' in checkpoint:
            ft_state = checkpoint['fishertune']
            self.current_dr_fim = {
                k: v.to(self.device) for k, v in ft_state['dr_fim'].items()
            }
            self.fim_history = ft_state['fim_history']
            self.param_scheduler.selection_history = ft_state['selection_history']
            self.current_epoch = ft_state.get('epoch', 0)

    def get_training_stats(self) -> Dict:
        """Get FisherTune training statistics."""
        stats = {
            'use_fishertune': self.use_fishertune,
            'current_epoch': self.current_epoch,
            'training_step': self.training_step
        }

        if self.use_fishertune:
            stats['scheduler_stats'] = self.param_scheduler.get_selection_stats()
            stats['fim_history'] = self.fim_history[-10:] if self.fim_history else []
            stats['lr'] = self.ft_optimizer.get_lr()

        return stats
