"""
FisherTune Optimizer Wrapper.

Implements selective parameter updates based on DR-FIM with:
- Different learning rates for shared vs specific parameters
- Different regularization strengths
- Parameter masking based on importance
- Natural gradient support
"""

import torch
import torch.nn as nn
from torch.optim import Optimizer
from typing import Dict, Set, Optional, List
import numpy as np


class FisherTuneOptimizer:
    """
    Wrapper optimizer that applies FisherTune parameter selection.
    """

    def __init__(
        self,
        model: nn.Module,
        base_optimizer_class: type = torch.optim.Adam,
        shared_lr: float = 0.0001,
        specific_lr: float = 0.001,
        shared_weight_decay: float = 1e-3,
        specific_weight_decay: float = 1e-5,
        use_natural_gradient: bool = False,
        fisher_damping: float = 1e-4,
        **base_optimizer_kwargs
    ):
        """
        Args:
            model: Neural network model
            base_optimizer_class: Base optimizer class (Adam, SGD, etc.)
            shared_lr: Learning rate for shared parameters
            specific_lr: Learning rate for domain-specific parameters
            shared_weight_decay: L2 regularization for shared parameters
            specific_weight_decay: L2 regularization for specific parameters
            use_natural_gradient: Whether to use natural gradient updates
            fisher_damping: Damping for natural gradient
            **base_optimizer_kwargs: Additional arguments for base optimizer
        """
        self.model = model
        self.base_optimizer_class = base_optimizer_class
        self.shared_lr = shared_lr
        self.specific_lr = specific_lr
        self.shared_weight_decay = shared_weight_decay
        self.specific_weight_decay = specific_weight_decay
        self.use_natural_gradient = use_natural_gradient
        self.fisher_damping = fisher_damping

        # Classify parameters
        self.shared_params = []
        self.specific_params = []
        self.shared_param_names = set()
        self.specific_param_names = set()
        self._classify_parameters()

        # Create separate parameter groups
        param_groups = self._create_param_groups()

        # Initialize base optimizer
        self.optimizer = base_optimizer_class(
            param_groups,
            **base_optimizer_kwargs
        )

        # Currently active parameters
        self.active_params = set()

        # Gradient mask (for selective updates)
        self.gradient_masks = {}

        # Fisher information (for natural gradient)
        self.fisher_info = {}

    def _classify_parameters(self):
        """Classify model parameters as shared or specific."""
        shared_keywords = ['share', 'shared', 'common', 'global']
        specific_keywords = ['specific', 'domain', '_list', 'dis_list', 'agg_list']

        for name, param in self.model.named_parameters():
            is_specific = any(kw in name.lower() for kw in specific_keywords)
            is_shared = any(kw in name.lower() for kw in shared_keywords)

            if is_specific and not is_shared:
                self.specific_params.append(param)
                self.specific_param_names.add(name)
            else:
                self.shared_params.append(param)
                self.shared_param_names.add(name)

    def _create_param_groups(self) -> List[Dict]:
        """Create parameter groups with different hyperparameters."""
        groups = []

        if self.shared_params:
            groups.append({
                'params': self.shared_params,
                'lr': self.shared_lr,
                'weight_decay': self.shared_weight_decay,
                'group_type': 'shared'
            })

        if self.specific_params:
            groups.append({
                'params': self.specific_params,
                'lr': self.specific_lr,
                'weight_decay': self.specific_weight_decay,
                'group_type': 'specific'
            })

        return groups

    def update_active_params(self, active_params: Set[str]):
        """
        Update which parameters are active for training.

        Args:
            active_params: Set of parameter names to be updated
        """
        self.active_params = active_params

        # Create gradient masks
        self.gradient_masks = {}
        for name, param in self.model.named_parameters():
            if name in active_params:
                self.gradient_masks[name] = torch.ones_like(param)
            else:
                self.gradient_masks[name] = torch.zeros_like(param)

    def update_fisher_info(self, fisher_info: Dict[str, torch.Tensor]):
        """
        Update Fisher information for natural gradient.

        Args:
            fisher_info: Dictionary of Fisher information per parameter
        """
        self.fisher_info = fisher_info

    def step(self, closure=None):
        """
        Perform optimization step with selective parameter updates.

        Args:
            closure: Closure for loss computation (optional)
        """
        # Apply gradient masks before step
        if self.active_params:
            self._apply_gradient_masks()

        # Apply natural gradient if enabled
        if self.use_natural_gradient and self.fisher_info:
            self._apply_natural_gradient()

        # Perform base optimizer step
        return self.optimizer.step(closure)

    def _apply_gradient_masks(self):
        """Apply gradient masks to implement selective updates."""
        for name, param in self.model.named_parameters():
            if param.grad is not None and name in self.gradient_masks:
                param.grad.data *= self.gradient_masks[name]

    def _apply_natural_gradient(self):
        """
        Apply natural gradient: g_natural = F^-1 * g

        Natural gradient accounts for parameter geometry.
        """
        for name, param in self.model.named_parameters():
            if param.grad is not None and name in self.fisher_info:
                fisher = self.fisher_info[name]
                # Natural gradient = gradient / (Fisher + damping)
                param.grad.data /= (fisher + self.fisher_damping)

    def zero_grad(self):
        """Zero out gradients."""
        self.optimizer.zero_grad()

    def state_dict(self):
        """Get optimizer state dict."""
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        """Load optimizer state dict."""
        self.optimizer.load_state_dict(state_dict)

    def get_lr(self) -> Dict[str, float]:
        """Get current learning rates."""
        return {
            'shared': self.shared_lr,
            'specific': self.specific_lr
        }

    def set_lr(self, shared_lr: Optional[float] = None, specific_lr: Optional[float] = None):
        """
        Set learning rates for parameter groups.

        Args:
            shared_lr: New learning rate for shared parameters
            specific_lr: New learning rate for specific parameters
        """
        if shared_lr is not None:
            self.shared_lr = shared_lr
        if specific_lr is not None:
            self.specific_lr = specific_lr

        # Update optimizer parameter groups
        for group in self.optimizer.param_groups:
            if group.get('group_type') == 'shared' and shared_lr is not None:
                group['lr'] = shared_lr
            elif group.get('group_type') == 'specific' and specific_lr is not None:
                group['lr'] = specific_lr

    def decay_lr(self, decay_factor: float = 0.98):
        """
        Decay learning rates by a factor.

        Args:
            decay_factor: Multiplicative decay factor
        """
        self.set_lr(
            shared_lr=self.shared_lr * decay_factor,
            specific_lr=self.specific_lr * decay_factor
        )


class FisherRegularizedOptimizer(FisherTuneOptimizer):
    """
    Optimizer with Fisher-based regularization.
    Penalizes updates to important parameters more heavily.
    """

    def __init__(
        self,
        model: nn.Module,
        base_optimizer_class: type = torch.optim.Adam,
        shared_lr: float = 0.0001,
        specific_lr: float = 0.001,
        shared_weight_decay: float = 1e-3,
        specific_weight_decay: float = 1e-5,
        fisher_reg_strength: float = 0.1,
        **base_optimizer_kwargs
    ):
        """
        Args:
            fisher_reg_strength: Strength of Fisher-based regularization
        """
        super().__init__(
            model, base_optimizer_class, shared_lr, specific_lr,
            shared_weight_decay, specific_weight_decay,
            **base_optimizer_kwargs
        )
        self.fisher_reg_strength = fisher_reg_strength

        # Store initial parameters for regularization
        self.initial_params = {}
        for name, param in model.named_parameters():
            self.initial_params[name] = param.data.clone()

    def compute_fisher_regularization(self) -> torch.Tensor:
        """
        Compute Fisher-weighted regularization loss.

        L_reg = sum_i F_i * (θ_i - θ_i^0)²

        Returns:
            Regularization loss
        """
        reg_loss = 0.0

        for name, param in self.model.named_parameters():
            if name in self.fisher_info and name in self.initial_params:
                # Fisher-weighted L2 distance from initial
                diff = param - self.initial_params[name]
                fisher_weight = self.fisher_info[name]
                reg_loss += (fisher_weight * diff ** 2).sum()

        return self.fisher_reg_strength * reg_loss

    def get_regularized_loss(self, base_loss: torch.Tensor) -> torch.Tensor:
        """
        Get total loss including Fisher regularization.

        Args:
            base_loss: Base task loss

        Returns:
            Regularized loss
        """
        reg_loss = self.compute_fisher_regularization()
        return base_loss + reg_loss


class ElasticWeightConsolidation(FisherTuneOptimizer):
    """
    Elastic Weight Consolidation (EWC) inspired optimizer.
    Protects important parameters while allowing less important ones to adapt.
    """

    def __init__(
        self,
        model: nn.Module,
        base_optimizer_class: type = torch.optim.Adam,
        shared_lr: float = 0.0001,
        specific_lr: float = 0.001,
        ewc_lambda: float = 1000.0,
        **base_optimizer_kwargs
    ):
        """
        Args:
            ewc_lambda: EWC regularization strength
        """
        super().__init__(
            model, base_optimizer_class, shared_lr, specific_lr,
            **base_optimizer_kwargs
        )
        self.ewc_lambda = ewc_lambda

        # Store reference parameters and their importance
        self.reference_params = {}
        self.importance_weights = {}

        # Initialize with current parameters
        self.consolidate()

    def consolidate(self):
        """
        Consolidate current parameters as reference.
        Should be called after training on each domain.
        """
        for name, param in self.model.named_parameters():
            self.reference_params[name] = param.data.clone()

            # Use Fisher info as importance if available
            if name in self.fisher_info:
                self.importance_weights[name] = self.fisher_info[name].clone()
            else:
                # Default importance
                self.importance_weights[name] = torch.ones_like(param)

    def compute_ewc_loss(self) -> torch.Tensor:
        """
        Compute EWC regularization loss.

        L_ewc = (λ/2) * sum_i F_i * (θ_i - θ_i^*)²

        Returns:
            EWC loss
        """
        ewc_loss = 0.0

        for name, param in self.model.named_parameters():
            if name in self.reference_params:
                diff = param - self.reference_params[name]
                importance = self.importance_weights.get(
                    name, torch.ones_like(param)
                )
                ewc_loss += (importance * diff ** 2).sum()

        return (self.ewc_lambda / 2.0) * ewc_loss


class LayerWiseLROptimizer(FisherTuneOptimizer):
    """
    Optimizer with layer-wise learning rate scaling.
    Deeper layers get smaller learning rates (transfer learning style).
    """

    def __init__(
        self,
        model: nn.Module,
        base_optimizer_class: type = torch.optim.Adam,
        base_lr: float = 0.001,
        layer_decay: float = 0.8,
        **base_optimizer_kwargs
    ):
        """
        Args:
            base_lr: Base learning rate (for top layers)
            layer_decay: Multiplicative decay per layer depth
        """
        self.base_lr = base_lr
        self.layer_decay = layer_decay

        # Assign layer-wise LRs
        param_groups = self._create_layerwise_groups(model)

        # Initialize base optimizer directly
        self.model = model
        self.optimizer = base_optimizer_class(
            param_groups,
            **base_optimizer_kwargs
        )

        self.active_params = set()
        self.gradient_masks = {}
        self.fisher_info = {}
        self.use_natural_gradient = False
        self.fisher_damping = 1e-4

        # Classify for reference
        self.shared_param_names = set()
        self.specific_param_names = set()
        self.shared_params = []
        self.specific_params = []

    def _create_layerwise_groups(self, model: nn.Module) -> List[Dict]:
        """Create parameter groups with layer-wise learning rates."""
        # Group parameters by estimated depth
        depth_params = {}

        for name, param in model.named_parameters():
            # Estimate depth from name
            depth = self._estimate_layer_depth(name)

            if depth not in depth_params:
                depth_params[depth] = []
            depth_params[depth].append(param)

        # Create groups with decaying LRs
        max_depth = max(depth_params.keys()) if depth_params else 0
        groups = []

        for depth, params in sorted(depth_params.items(), reverse=True):
            # Higher depth (closer to output) = higher LR
            depth_factor = (max_depth - depth) / max(max_depth, 1)
            lr = self.base_lr * (self.layer_decay ** depth_factor)

            groups.append({
                'params': params,
                'lr': lr,
                'depth': depth
            })

        return groups

    def _estimate_layer_depth(self, param_name: str) -> int:
        """Estimate layer depth from parameter name."""
        # Look for numeric indices in name
        parts = param_name.split('.')
        depth = 0

        for part in parts:
            if part.isdigit():
                depth = max(depth, int(part))

        return depth


def create_optimizer(
    model: nn.Module,
    optimizer_type: str = 'fishertune',
    **kwargs
) -> FisherTuneOptimizer:
    """
    Factory function to create FisherTune optimizer.

    Args:
        model: Neural network
        optimizer_type: Type of optimizer
            - 'fishertune': Basic FisherTune optimizer
            - 'fisher_reg': Fisher-regularized optimizer
            - 'ewc': Elastic Weight Consolidation
            - 'layerwise': Layer-wise learning rates
        **kwargs: Optimizer arguments

    Returns:
        FisherTune optimizer instance
    """
    if optimizer_type == 'fishertune':
        return FisherTuneOptimizer(model, **kwargs)
    elif optimizer_type == 'fisher_reg':
        return FisherRegularizedOptimizer(model, **kwargs)
    elif optimizer_type == 'ewc':
        return ElasticWeightConsolidation(model, **kwargs)
    elif optimizer_type == 'layerwise':
        return LayerWiseLROptimizer(model, **kwargs)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
