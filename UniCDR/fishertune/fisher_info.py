"""
Fisher Information Matrix computation for CDR
Implements diagonal FIM approximation and Domain-Related FIM (DR-FIM)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import numpy as np
from collections import defaultdict


class FisherInformationComputer:
    """
    Computes Fisher Information Matrix using diagonal approximation.

    FIM measures parameter sensitivity: F_θ = E[(∇_θ L)²]
    High FIM value = parameter is important for the task
    """

    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.fim_cache = {}
        self.accumulated_fim = {}
        self.num_samples = 0

    def compute_diagonal_fim(
        self,
        loss: torch.Tensor,
        retain_graph: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Compute diagonal FIM from a single loss value.
        F_θ,n ≈ (∂L/∂θ_n)²

        Args:
            loss: Scalar loss value
            retain_graph: Whether to retain computation graph

        Returns:
            Dictionary mapping parameter name to FIM diagonal
        """
        fim_dict = {}

        # Compute gradients
        grads = torch.autograd.grad(
            loss,
            self.model.parameters(),
            create_graph=retain_graph,
            retain_graph=retain_graph,
            allow_unused=True
        )

        for (name, param), grad in zip(self.model.named_parameters(), grads):
            if grad is not None:
                # Diagonal FIM: squared gradients
                fim_dict[name] = (grad ** 2).detach()
            else:
                fim_dict[name] = torch.zeros_like(param)

        return fim_dict

    def accumulate_fim(
        self,
        loss: torch.Tensor,
        retain_graph: bool = False
    ):
        """
        Accumulate FIM estimates across multiple samples.
        Online mean update: F_new = F_old + (F_sample - F_old) / n
        """
        current_fim = self.compute_diagonal_fim(loss, retain_graph)
        self.num_samples += 1

        for name, fim_val in current_fim.items():
            if name not in self.accumulated_fim:
                self.accumulated_fim[name] = fim_val.clone()
            else:
                # Online mean update for stability
                delta = fim_val - self.accumulated_fim[name]
                self.accumulated_fim[name] += delta / self.num_samples

    def get_accumulated_fim(self) -> Dict[str, torch.Tensor]:
        """Get the accumulated FIM estimates."""
        return self.accumulated_fim.copy()

    def reset_accumulation(self):
        """Reset accumulated FIM."""
        self.accumulated_fim = {}
        self.num_samples = 0

    def compute_domain_fim(
        self,
        domain_batches: List[Tuple],
        loss_fn: callable,
        num_samples: int = 100
    ) -> Dict[str, torch.Tensor]:
        """
        Compute FIM for a specific domain.

        Args:
            domain_batches: Batches from the domain
            loss_fn: Function to compute loss given batch
            num_samples: Number of samples to accumulate

        Returns:
            Domain-specific FIM
        """
        self.reset_accumulation()

        for i, batch in enumerate(domain_batches):
            if i >= num_samples:
                break

            loss = loss_fn(batch)
            self.accumulate_fim(loss, retain_graph=False)

        return self.get_accumulated_fim()


class DomainRelatedFIM:
    """
    Computes Domain-Related Fisher Information Matrix (DR-FIM).

    DR-FIM = F_θ(x,y) + exp(-ε) * ΔF_θ

    Where ΔF_θ measures how FIM changes across domains.
    """

    def __init__(self, epsilon: float = 1e-8):
        self.epsilon = epsilon

    def compute_delta_fim(
        self,
        fim_source: Dict[str, torch.Tensor],
        fim_perturbed: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute normalized FIM change between domains.

        ΔF_θ = |F_θ(x) - F_θ(x')| / (min(F_θ(x), F_θ(x')) + ε)

        Args:
            fim_source: FIM from source domain
            fim_perturbed: FIM from perturbed/target domain

        Returns:
            Normalized FIM difference
        """
        delta_fim = {}

        for name in fim_source.keys():
            if name in fim_perturbed:
                diff = torch.abs(fim_source[name] - fim_perturbed[name])
                min_val = torch.minimum(fim_source[name], fim_perturbed[name])
                delta_fim[name] = diff / (min_val + self.epsilon)
            else:
                delta_fim[name] = torch.zeros_like(fim_source[name])

        return delta_fim

    def compute_dr_fim(
        self,
        fim_source: Dict[str, torch.Tensor],
        delta_fim: Dict[str, torch.Tensor],
        perturbation_noise: float = 0.1
    ) -> Dict[str, torch.Tensor]:
        """
        Compute Domain-Related FIM.

        DR-FIM = F_θ(x,y) + exp(-(ε_μ + ε_σ)) * ΔF_θ

        Args:
            fim_source: Source domain FIM
            delta_fim: FIM change
            perturbation_noise: Combined perturbation noise (ε_μ + ε_σ)

        Returns:
            DR-FIM values
        """
        dr_fim = {}
        weight = np.exp(-perturbation_noise)

        for name in fim_source.keys():
            if name in delta_fim:
                dr_fim[name] = fim_source[name] + weight * delta_fim[name]
            else:
                dr_fim[name] = fim_source[name]

        return dr_fim


class MultiDomainFIM:
    """
    Manages FIM computation across multiple domains in CDR setting.
    """

    def __init__(
        self,
        model: nn.Module,
        num_domains: int,
        device: str = 'cuda'
    ):
        self.model = model
        self.num_domains = num_domains
        self.device = device
        self.fim_computer = FisherInformationComputer(model, device)
        self.dr_fim_computer = DomainRelatedFIM()

        # Store FIM per domain
        self.domain_fims = {}
        self.perturbed_fims = {}

    def compute_all_domain_fims(
        self,
        domain_loaders: List,
        loss_fn: callable,
        num_samples_per_domain: int = 100
    ):
        """
        Compute FIM for all domains.
        """
        for domain_id in range(self.num_domains):
            self.domain_fims[domain_id] = self.fim_computer.compute_domain_fim(
                domain_loaders[domain_id],
                lambda batch: loss_fn(domain_id, batch),
                num_samples_per_domain
            )

    def compute_cross_domain_dr_fim(
        self,
        source_domain: int,
        target_domain: int,
        perturbation_noise: float = 0.1
    ) -> Dict[str, torch.Tensor]:
        """
        Compute DR-FIM using actual domain pairs (for CDR setting).
        Since CDR has multiple domains, we can use actual domain pairs
        instead of simulated perturbations.
        """
        if source_domain not in self.domain_fims:
            raise ValueError(f"FIM for domain {source_domain} not computed")
        if target_domain not in self.domain_fims:
            raise ValueError(f"FIM for domain {target_domain} not computed")

        delta_fim = self.dr_fim_computer.compute_delta_fim(
            self.domain_fims[source_domain],
            self.domain_fims[target_domain]
        )

        return self.dr_fim_computer.compute_dr_fim(
            self.domain_fims[source_domain],
            delta_fim,
            perturbation_noise
        )

    def compute_aggregated_dr_fim(
        self,
        perturbation_noise: float = 0.1
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate DR-FIM across all domain pairs.
        For CDR, we consider all domain combinations.
        """
        aggregated = {}
        num_pairs = 0

        for src in range(self.num_domains):
            for tgt in range(self.num_domains):
                if src != tgt:
                    dr_fim = self.compute_cross_domain_dr_fim(
                        src, tgt, perturbation_noise
                    )

                    for name, val in dr_fim.items():
                        if name not in aggregated:
                            aggregated[name] = val.clone()
                        else:
                            aggregated[name] += val
                    num_pairs += 1

        # Average
        if num_pairs > 0:
            for name in aggregated:
                aggregated[name] /= num_pairs

        return aggregated


def identify_parameter_type(param_name: str) -> str:
    """
    Identify if parameter is shared or domain-specific.

    Args:
        param_name: Name of the parameter

    Returns:
        'shared' or 'specific'
    """
    shared_keywords = ['share', 'shared', 'common', 'global']
    specific_keywords = ['specific', 'domain', '_list', 'dis_list', 'agg_list']

    # Check for specific patterns first (they're more distinctive)
    for keyword in specific_keywords:
        if keyword in param_name.lower():
            return 'specific'

    for keyword in shared_keywords:
        if keyword in param_name.lower():
            return 'shared'

    # Default to shared for common components
    return 'shared'


def separate_fim_by_type(
    fim_dict: Dict[str, torch.Tensor]
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Separate FIM dictionary into shared and specific parameters.

    Args:
        fim_dict: Complete FIM dictionary

    Returns:
        (shared_fim, specific_fim)
    """
    shared_fim = {}
    specific_fim = {}

    for name, fim_val in fim_dict.items():
        if identify_parameter_type(name) == 'shared':
            shared_fim[name] = fim_val
        else:
            specific_fim[name] = fim_val

    return shared_fim, specific_fim
