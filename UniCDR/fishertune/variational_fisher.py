"""
Variational Inference based stable Fisher Information estimation.

Uses variational approximation to stabilize FIM estimation:
- Posterior: q(θ) = N(θ_hat, Λ^-1)
- Prior: p(θ) = N(θ_pt, τ²I)
- Objective: L_VI = E_q[L(θ)] + γ * KL(q||p)

The precision matrix Λ relates to stabilized Fisher:
F_θ = γ(Λ - τ^-2 I)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import numpy as np


class VariationalFisher:
    """
    Variational inference based Fisher Information estimation.
    Provides more stable and regularized FIM estimates.
    """

    def __init__(
        self,
        model: nn.Module,
        tau: float = 1.0,
        gamma: float = 0.1,
        num_samples: int = 1,
        device: str = 'cuda'
    ):
        """
        Args:
            model: Neural network model
            tau: Prior standard deviation (τ²)
            gamma: KL divergence weight
            num_samples: Number of samples for ELBO estimation
            device: Computation device
        """
        self.model = model
        self.tau = tau
        self.tau_sq = tau ** 2
        self.gamma = gamma
        self.num_samples = num_samples
        self.device = device

        # Store initial/pretrained parameters as prior mean
        self.prior_mean = {}
        for name, param in model.named_parameters():
            self.prior_mean[name] = param.data.clone()

        # Initialize variational parameters (posterior precision)
        # Λ = precision = 1/variance
        # Start with relatively small variance (high precision)
        self.lambda_precision = {}
        for name, param in model.named_parameters():
            # Initialize precision close to prior (1/τ²)
            initial_precision = torch.ones_like(param) / self.tau_sq
            self.lambda_precision[name] = initial_precision.to(device)

    def compute_kl_divergence(self) -> torch.Tensor:
        """
        Compute KL divergence between posterior and prior.

        KL(q||p) = 0.5 * sum(
            Λ/τ^-2 - log(Λτ²) - 1 + (θ_hat - θ_pt)²/τ²
        )

        For diagonal case.
        """
        kl_div = 0.0

        for name, param in self.model.named_parameters():
            if name in self.lambda_precision:
                lambda_p = self.lambda_precision[name]
                theta_diff = param - self.prior_mean[name]

                # KL terms for diagonal Gaussian
                # Variance ratio term
                var_ratio = lambda_p * self.tau_sq

                # Log determinant term (for diagonal, sum of logs)
                log_det = torch.log(var_ratio + 1e-8)

                # Mean difference term
                mean_term = (theta_diff ** 2) / self.tau_sq

                # Sum for this parameter group
                kl_param = 0.5 * (var_ratio - log_det - 1 + mean_term).sum()
                kl_div += kl_param

        return kl_div

    def compute_variational_loss(
        self,
        data_loss: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute variational objective:
        L_VI = E_q[L(θ)] + γ * KL(q||p)

        Args:
            data_loss: Loss from data likelihood

        Returns:
            Total variational loss
        """
        kl_loss = self.compute_kl_divergence()
        return data_loss + self.gamma * kl_loss

    def update_precision_from_loss(
        self,
        loss: torch.Tensor,
        learning_rate: float = 0.01
    ):
        """
        Update precision estimates from loss gradients.

        Using the relationship: F_θ ≈ Hessian ≈ γ(Λ - τ^-2 I)
        We can update Λ based on squared gradients.
        """
        # Compute gradients
        grads = torch.autograd.grad(
            loss,
            self.model.parameters(),
            create_graph=False,
            retain_graph=True,
            allow_unused=True
        )

        for (name, param), grad in zip(self.model.named_parameters(), grads):
            if grad is not None and name in self.lambda_precision:
                # Update precision: higher gradient -> higher Fisher -> higher precision
                # λ_new = λ_old + lr * (g² / γ + τ^-2)
                grad_sq = grad.detach() ** 2
                fisher_estimate = grad_sq / self.gamma + 1.0 / self.tau_sq

                # Exponential moving average update
                self.lambda_precision[name] = (
                    (1 - learning_rate) * self.lambda_precision[name] +
                    learning_rate * fisher_estimate
                )

    def get_stabilized_fisher(self) -> Dict[str, torch.Tensor]:
        """
        Get stabilized Fisher Information from precision matrix.

        F_θ = γ(Λ - τ^-2 I)

        Returns:
            Dictionary of stabilized FIM values
        """
        stabilized_fim = {}

        for name in self.lambda_precision:
            # F = γ(Λ - 1/τ²)
            fisher_val = self.gamma * (
                self.lambda_precision[name] - 1.0 / self.tau_sq
            )
            # Ensure non-negative
            stabilized_fim[name] = torch.clamp(fisher_val, min=0)

        return stabilized_fim

    def get_stabilized_dr_fim(
        self,
        perturbed_precision: Dict[str, torch.Tensor],
        perturbation_noise: float = 0.1
    ) -> Dict[str, torch.Tensor]:
        """
        Compute stabilized Domain-Related FIM.

        DR-FIM = γ(Λ_x - τ^-2 I + exp(-(ε_μ+ε_σ)) * |Λ_x - Λ_x'| / (min(Λ_x, Λ_x') + ε) / γ)

        Args:
            perturbed_precision: Precision from perturbed domain
            perturbation_noise: Combined perturbation magnitude

        Returns:
            Stabilized DR-FIM
        """
        dr_fim = {}
        weight = np.exp(-perturbation_noise) / self.gamma
        epsilon = 1e-8

        for name in self.lambda_precision:
            if name in perturbed_precision:
                lambda_x = self.lambda_precision[name]
                lambda_xp = perturbed_precision[name]

                # Base Fisher
                base_fisher = self.gamma * (lambda_x - 1.0 / self.tau_sq)

                # Delta term
                delta = torch.abs(lambda_x - lambda_xp)
                min_lambda = torch.minimum(lambda_x, lambda_xp)
                delta_normalized = delta / (min_lambda + epsilon)

                # Combined DR-FIM
                dr_fim[name] = torch.clamp(
                    base_fisher + weight * delta_normalized,
                    min=0
                )
            else:
                dr_fim[name] = torch.clamp(
                    self.gamma * (self.lambda_precision[name] - 1.0 / self.tau_sq),
                    min=0
                )

        return dr_fim

    def sample_parameters(self) -> Dict[str, torch.Tensor]:
        """
        Sample parameters from variational posterior.

        θ ~ N(θ_hat, Λ^-1)

        Returns:
            Dictionary of sampled parameters
        """
        sampled = {}

        for name, param in self.model.named_parameters():
            if name in self.lambda_precision:
                # Variance = 1/precision
                std = 1.0 / torch.sqrt(self.lambda_precision[name] + 1e-8)
                noise = torch.randn_like(param)
                sampled[name] = param + std * noise
            else:
                sampled[name] = param.clone()

        return sampled


class AdaptiveVariationalFisher(VariationalFisher):
    """
    Adaptive version that adjusts tau and gamma during training.
    """

    def __init__(
        self,
        model: nn.Module,
        tau_init: float = 1.0,
        gamma_init: float = 0.1,
        tau_decay: float = 0.99,
        gamma_growth: float = 1.01,
        min_tau: float = 0.1,
        max_gamma: float = 1.0,
        device: str = 'cuda'
    ):
        super().__init__(model, tau_init, gamma_init, device=device)
        self.tau_decay = tau_decay
        self.gamma_growth = gamma_growth
        self.min_tau = min_tau
        self.max_gamma = max_gamma

    def adapt_hyperparameters(self):
        """
        Adapt tau and gamma based on training progress.
        - Decrease tau: tighten prior as training progresses
        - Increase gamma: stronger regularization over time
        """
        # Decay tau (tighter prior)
        self.tau = max(self.min_tau, self.tau * self.tau_decay)
        self.tau_sq = self.tau ** 2

        # Grow gamma (stronger KL regularization)
        self.gamma = min(self.max_gamma, self.gamma * self.gamma_growth)


class OnlineFisherEstimator:
    """
    Online estimation of Fisher Information with momentum.
    More memory efficient than storing all samples.
    """

    def __init__(
        self,
        model: nn.Module,
        momentum: float = 0.9,
        damping: float = 1e-4,
        device: str = 'cuda'
    ):
        """
        Args:
            model: Neural network
            momentum: EMA momentum (higher = more stable, slower adaptation)
            damping: Damping term for numerical stability
            device: Computation device
        """
        self.model = model
        self.momentum = momentum
        self.damping = damping
        self.device = device

        # Running Fisher estimate (diagonal)
        self.running_fisher = {}
        for name, param in model.named_parameters():
            self.running_fisher[name] = torch.ones_like(param) * damping

        self.num_updates = 0

    def update(self, loss: torch.Tensor):
        """
        Update running Fisher estimate with current loss.

        F_new = momentum * F_old + (1 - momentum) * g²
        """
        grads = torch.autograd.grad(
            loss,
            self.model.parameters(),
            create_graph=False,
            retain_graph=True,
            allow_unused=True
        )

        for (name, param), grad in zip(self.model.named_parameters(), grads):
            if grad is not None:
                grad_sq = grad.detach() ** 2

                if self.num_updates == 0:
                    self.running_fisher[name] = grad_sq + self.damping
                else:
                    self.running_fisher[name] = (
                        self.momentum * self.running_fisher[name] +
                        (1 - self.momentum) * grad_sq
                    )

        self.num_updates += 1

    def get_fisher(self) -> Dict[str, torch.Tensor]:
        """Get current Fisher estimates."""
        return self.running_fisher.copy()

    def get_natural_gradient(
        self,
        gradients: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute natural gradient: F^-1 * g

        Natural gradient accounts for parameter geometry.
        """
        natural_grads = {}

        for name, grad in gradients.items():
            if name in self.running_fisher:
                # Natural gradient = gradient / Fisher
                natural_grads[name] = grad / (self.running_fisher[name] + self.damping)
            else:
                natural_grads[name] = grad

        return natural_grads
