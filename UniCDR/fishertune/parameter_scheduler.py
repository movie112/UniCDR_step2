"""
Parameter Selection Scheduler for FisherTune.

Implements progressive parameter selection based on DR-FIM values.
Key idea: Start with most important (high DR-FIM) parameters, gradually expand.

Threshold scheduling:
DR-F_thresh(t) = δ_min + (δ_max - δ_min) * exp(-t/T)

Parameters with DR-FIM > threshold are selected for tuning.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Set, Optional, Tuple
import numpy as np
from collections import defaultdict


class ParameterScheduler:
    """
    Schedules which parameters to update based on DR-FIM importance.
    """

    def __init__(
        self,
        model: nn.Module,
        delta_min: float = 0.1,
        delta_max: float = 0.9,
        decay_constant: float = 50.0,
        warmup_epochs: int = 5,
        device: str = 'cuda'
    ):
        """
        Args:
            model: Neural network model
            delta_min: Minimum threshold (final value, more parameters)
            delta_max: Maximum threshold (initial value, fewer parameters)
            decay_constant: Time constant for exponential decay (T)
            warmup_epochs: Epochs before applying FisherTune
            device: Computation device
        """
        self.model = model
        self.delta_min = delta_min
        self.delta_max = delta_max
        self.decay_constant = decay_constant
        self.warmup_epochs = warmup_epochs
        self.device = device

        # Track which parameters are currently active
        self.active_params = set()

        # Store DR-FIM values
        self.dr_fim_values = {}

        # Track parameter selection history
        self.selection_history = []

        # Current threshold
        self.current_threshold = delta_max

        # Total number of parameters
        self.total_params = sum(p.numel() for p in model.parameters())

    def compute_threshold(self, epoch: int) -> float:
        """
        Compute threshold for current epoch.

        δ(t) = δ_min + (δ_max - δ_min) * exp(-t/T)

        Args:
            epoch: Current training epoch

        Returns:
            Current threshold value
        """
        if epoch < self.warmup_epochs:
            # During warmup, use maximum threshold (minimal tuning)
            return self.delta_max

        # Adjusted epoch (after warmup)
        t = epoch - self.warmup_epochs

        # Exponential decay
        threshold = self.delta_min + (self.delta_max - self.delta_min) * \
                    np.exp(-t / self.decay_constant)

        self.current_threshold = threshold
        return threshold

    def update_dr_fim(self, dr_fim: Dict[str, torch.Tensor]):
        """
        Update stored DR-FIM values.

        Args:
            dr_fim: Dictionary of DR-FIM values per parameter
        """
        self.dr_fim_values = dr_fim

    def select_parameters(
        self,
        epoch: int,
        percentile_based: bool = True
    ) -> Set[str]:
        """
        Select parameters to update based on DR-FIM and threshold.

        Args:
            epoch: Current epoch
            percentile_based: Use percentile-based selection instead of absolute

        Returns:
            Set of parameter names to update
        """
        if not self.dr_fim_values:
            # If no DR-FIM computed yet, update all parameters
            return set(name for name, _ in self.model.named_parameters())

        threshold = self.compute_threshold(epoch)

        if percentile_based:
            return self._select_by_percentile(threshold)
        else:
            return self._select_by_absolute(threshold)

    def _select_by_percentile(self, threshold: float) -> Set[str]:
        """
        Select parameters by percentile ranking.

        threshold determines what percentile of parameters to select.
        threshold=0.9 -> top 10% parameters
        threshold=0.1 -> top 90% parameters
        """
        # Flatten all DR-FIM values
        all_values = []
        param_fim_map = {}

        for name, fim_tensor in self.dr_fim_values.items():
            mean_fim = fim_tensor.mean().item()
            all_values.append(mean_fim)
            param_fim_map[name] = mean_fim

        if not all_values:
            return set()

        # Compute percentile threshold
        percentile_value = np.percentile(all_values, threshold * 100)

        # Select parameters above percentile
        selected = set()
        for name, fim_val in param_fim_map.items():
            if fim_val >= percentile_value:
                selected.add(name)

        self.active_params = selected
        self._log_selection(threshold, len(selected))

        return selected

    def _select_by_absolute(self, threshold: float) -> Set[str]:
        """
        Select parameters by absolute DR-FIM value.
        """
        selected = set()

        for name, fim_tensor in self.dr_fim_values.items():
            # Use mean FIM value for parameter group
            mean_fim = fim_tensor.mean().item()

            if mean_fim >= threshold:
                selected.add(name)

        self.active_params = selected
        self._log_selection(threshold, len(selected))

        return selected

    def _log_selection(self, threshold: float, num_selected: int):
        """Log parameter selection info."""
        total_param_groups = len(list(self.model.named_parameters()))
        selection_ratio = num_selected / max(total_param_groups, 1)

        self.selection_history.append({
            'threshold': threshold,
            'num_selected': num_selected,
            'total_groups': total_param_groups,
            'selection_ratio': selection_ratio
        })

    def get_parameter_mask(
        self,
        selected_params: Set[str]
    ) -> Dict[str, torch.Tensor]:
        """
        Create binary masks for parameter selection.

        Args:
            selected_params: Set of parameter names to update

        Returns:
            Dictionary of binary masks
        """
        masks = {}

        for name, param in self.model.named_parameters():
            if name in selected_params:
                masks[name] = torch.ones_like(param)
            else:
                masks[name] = torch.zeros_like(param)

        return masks

    def get_selection_stats(self) -> Dict:
        """Get statistics about current parameter selection."""
        if not self.selection_history:
            return {}

        latest = self.selection_history[-1]

        return {
            'current_threshold': self.current_threshold,
            'num_active_params': latest['num_selected'],
            'selection_ratio': latest['selection_ratio'],
            'total_updates': len(self.selection_history)
        }


class SharedSpecificScheduler(ParameterScheduler):
    """
    Scheduler with different thresholds for shared vs specific parameters.

    Shared: High δ (mostly frozen for invariance)
    Specific: Low δ (more tuning allowed)
    """

    def __init__(
        self,
        model: nn.Module,
        shared_delta_min: float = 0.5,
        shared_delta_max: float = 0.95,
        specific_delta_min: float = 0.1,
        specific_delta_max: float = 0.7,
        decay_constant: float = 50.0,
        warmup_epochs: int = 5,
        device: str = 'cuda'
    ):
        """
        Args:
            model: Neural network
            shared_delta_min/max: Threshold range for shared parameters
            specific_delta_min/max: Threshold range for specific parameters
            decay_constant: Decay time constant
            warmup_epochs: Warmup period
            device: Computation device
        """
        super().__init__(
            model, shared_delta_min, shared_delta_max,
            decay_constant, warmup_epochs, device
        )

        self.shared_delta_min = shared_delta_min
        self.shared_delta_max = shared_delta_max
        self.specific_delta_min = specific_delta_min
        self.specific_delta_max = specific_delta_max

        # Classify parameters
        self.shared_params = set()
        self.specific_params = set()
        self._classify_parameters()

    def _classify_parameters(self):
        """Classify parameters as shared or specific."""
        shared_keywords = ['share', 'shared', 'common', 'global']
        specific_keywords = ['specific', 'domain', '_list', 'dis_list', 'agg_list']

        for name, _ in self.model.named_parameters():
            is_specific = any(kw in name.lower() for kw in specific_keywords)
            is_shared = any(kw in name.lower() for kw in shared_keywords)

            if is_specific and not is_shared:
                self.specific_params.add(name)
            else:
                self.shared_params.add(name)

    def compute_thresholds(self, epoch: int) -> Tuple[float, float]:
        """
        Compute separate thresholds for shared and specific parameters.

        Returns:
            (shared_threshold, specific_threshold)
        """
        if epoch < self.warmup_epochs:
            return self.shared_delta_max, self.specific_delta_max

        t = epoch - self.warmup_epochs

        shared_thresh = self.shared_delta_min + \
                       (self.shared_delta_max - self.shared_delta_min) * \
                       np.exp(-t / self.decay_constant)

        specific_thresh = self.specific_delta_min + \
                         (self.specific_delta_max - self.specific_delta_min) * \
                         np.exp(-t / self.decay_constant)

        return shared_thresh, specific_thresh

    def select_parameters(
        self,
        epoch: int,
        percentile_based: bool = True
    ) -> Set[str]:
        """
        Select parameters with different thresholds for shared/specific.
        """
        if not self.dr_fim_values:
            return set(name for name, _ in self.model.named_parameters())

        shared_thresh, specific_thresh = self.compute_thresholds(epoch)

        selected = set()

        # Process shared parameters
        shared_fims = {
            name: self.dr_fim_values[name]
            for name in self.shared_params
            if name in self.dr_fim_values
        }
        selected.update(
            self._select_from_subset(shared_fims, shared_thresh, percentile_based)
        )

        # Process specific parameters
        specific_fims = {
            name: self.dr_fim_values[name]
            for name in self.specific_params
            if name in self.dr_fim_values
        }
        selected.update(
            self._select_from_subset(specific_fims, specific_thresh, percentile_based)
        )

        self.active_params = selected
        self._log_selection(
            (shared_thresh + specific_thresh) / 2,
            len(selected)
        )

        return selected

    def _select_from_subset(
        self,
        fim_subset: Dict[str, torch.Tensor],
        threshold: float,
        percentile_based: bool
    ) -> Set[str]:
        """Select parameters from a subset."""
        if not fim_subset:
            return set()

        if percentile_based:
            all_values = [v.mean().item() for v in fim_subset.values()]
            if not all_values:
                return set()

            percentile_value = np.percentile(all_values, threshold * 100)

            return {
                name for name, fim in fim_subset.items()
                if fim.mean().item() >= percentile_value
            }
        else:
            return {
                name for name, fim in fim_subset.items()
                if fim.mean().item() >= threshold
            }


class GradualUnfreezeScheduler:
    """
    Alternative scheduler: Gradually unfreeze parameters layer by layer.
    Starts from output layers, moves toward input layers.
    """

    def __init__(
        self,
        model: nn.Module,
        num_stages: int = 5,
        epochs_per_stage: int = 10,
        warmup_epochs: int = 5
    ):
        """
        Args:
            model: Neural network
            num_stages: Number of unfreezing stages
            epochs_per_stage: Epochs between stage transitions
            warmup_epochs: Initial warmup
        """
        self.model = model
        self.num_stages = num_stages
        self.epochs_per_stage = epochs_per_stage
        self.warmup_epochs = warmup_epochs

        # Group parameters by layer depth
        self.layer_groups = self._group_by_depth()

    def _group_by_depth(self) -> List[List[str]]:
        """
        Group parameters by their depth in the network.
        Later groups = closer to output.
        """
        # Simple heuristic: group by parameter name patterns
        groups = defaultdict(list)

        for name, _ in self.model.named_parameters():
            # Extract layer index from name
            parts = name.split('.')
            if len(parts) >= 2:
                # Use first index as depth indicator
                try:
                    idx = int(parts[1]) if parts[1].isdigit() else 0
                except:
                    idx = 0
                groups[idx].append(name)
            else:
                groups[0].append(name)

        # Sort by depth and return as list
        sorted_groups = [
            groups[k] for k in sorted(groups.keys(), reverse=True)
        ]

        return sorted_groups

    def get_unfrozen_params(self, epoch: int) -> Set[str]:
        """
        Get set of parameters that should be unfrozen at current epoch.
        """
        if epoch < self.warmup_epochs:
            # During warmup, only train output layers
            return set(self.layer_groups[0]) if self.layer_groups else set()

        # Determine current stage
        adjusted_epoch = epoch - self.warmup_epochs
        stage = min(
            adjusted_epoch // self.epochs_per_stage,
            len(self.layer_groups) - 1
        )

        # Unfreeze all layers up to current stage
        unfrozen = set()
        for i in range(stage + 1):
            if i < len(self.layer_groups):
                unfrozen.update(self.layer_groups[i])

        return unfrozen


class AdaptiveScheduler(SharedSpecificScheduler):
    """
    Adaptive scheduler that adjusts thresholds based on training dynamics.
    """

    def __init__(
        self,
        model: nn.Module,
        shared_delta_min: float = 0.5,
        shared_delta_max: float = 0.95,
        specific_delta_min: float = 0.1,
        specific_delta_max: float = 0.7,
        decay_constant: float = 50.0,
        warmup_epochs: int = 5,
        adaptation_rate: float = 0.1,
        device: str = 'cuda'
    ):
        super().__init__(
            model, shared_delta_min, shared_delta_max,
            specific_delta_min, specific_delta_max,
            decay_constant, warmup_epochs, device
        )
        self.adaptation_rate = adaptation_rate
        self.loss_history = []
        self.grad_norm_history = []

    def adapt_thresholds(
        self,
        current_loss: float,
        grad_norm: float
    ):
        """
        Adapt thresholds based on training dynamics.

        - If loss is plateauing: increase tuning (lower threshold)
        - If gradient norm is high: be more conservative (higher threshold)
        """
        self.loss_history.append(current_loss)
        self.grad_norm_history.append(grad_norm)

        if len(self.loss_history) < 10:
            return

        # Check loss improvement
        recent_loss = np.mean(self.loss_history[-5:])
        older_loss = np.mean(self.loss_history[-10:-5])

        loss_improvement = (older_loss - recent_loss) / (older_loss + 1e-8)

        # Adapt thresholds
        if loss_improvement < 0.01:  # Loss plateauing
            # Lower thresholds to allow more tuning
            adaptation = -self.adaptation_rate
        elif grad_norm > np.mean(self.grad_norm_history) * 1.5:
            # High gradient norm, be more conservative
            adaptation = self.adaptation_rate
        else:
            adaptation = 0

        # Apply adaptation
        self.shared_delta_min = np.clip(
            self.shared_delta_min + adaptation, 0.3, 0.8
        )
        self.specific_delta_min = np.clip(
            self.specific_delta_min + adaptation, 0.05, 0.5
        )
