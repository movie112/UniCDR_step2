"""
Domain perturbation strategies for simulating unseen/perturbed domains.
Used for computing domain-related Fisher Information.

Strategies:
1. Edge Dropout: Randomly remove edges from interaction graph
2. Popularity-based Edge Weighting: Reweight edges based on item popularity
3. Feature Noise Injection: Add Gaussian noise to embeddings
4. Interaction Masking: Mask subset of user-item interactions
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
import scipy.sparse as sp


class DomainPerturbation:
    """
    Main class for domain perturbation strategies.
    Simulates domain shift for DR-FIM computation.
    """

    def __init__(
        self,
        perturbation_type: str = 'combined',
        edge_dropout_rate: float = 0.2,
        noise_std: float = 0.1,
        popularity_alpha: float = 0.5,
        device: str = 'cuda'
    ):
        """
        Args:
            perturbation_type: Type of perturbation
                - 'edge_dropout': Random edge removal
                - 'popularity': Popularity-based reweighting
                - 'noise': Feature noise injection
                - 'mask': Interaction masking
                - 'combined': Multiple strategies combined
            edge_dropout_rate: Fraction of edges to drop
            noise_std: Standard deviation for noise injection
            popularity_alpha: Weight for popularity bias
            device: Computation device
        """
        self.perturbation_type = perturbation_type
        self.edge_dropout_rate = edge_dropout_rate
        self.noise_std = noise_std
        self.popularity_alpha = popularity_alpha
        self.device = device

    def perturb_batch(
        self,
        batch: Tuple,
        item_popularity: Optional[torch.Tensor] = None
    ) -> Tuple:
        """
        Apply perturbation to a batch of data.

        Args:
            batch: (user, pos_item, neg_item, context_items, context_scores,
                   global_items, global_scores)
            item_popularity: Item popularity scores

        Returns:
            Perturbed batch
        """
        if self.perturbation_type == 'edge_dropout':
            return self._edge_dropout_batch(batch)
        elif self.perturbation_type == 'popularity':
            return self._popularity_weight_batch(batch, item_popularity)
        elif self.perturbation_type == 'noise':
            return self._noise_injection_batch(batch)
        elif self.perturbation_type == 'mask':
            return self._interaction_mask_batch(batch)
        elif self.perturbation_type == 'combined':
            return self._combined_perturbation_batch(batch, item_popularity)
        else:
            return batch

    def _edge_dropout_batch(self, batch: Tuple) -> Tuple:
        """
        Apply edge dropout to context items.
        Simulates missing interactions in perturbed domain.
        """
        user, pos_item, neg_item, context_items, context_scores, \
            global_items, global_scores = batch

        # Dropout context items
        if context_items is not None:
            mask = torch.rand_like(context_items.float()) > self.edge_dropout_rate
            # Keep at least one item
            mask[:, 0] = True
            perturbed_context = context_items * mask.long()
            perturbed_scores = context_scores * mask.float()
        else:
            perturbed_context = context_items
            perturbed_scores = context_scores

        return (user, pos_item, neg_item, perturbed_context, perturbed_scores,
                global_items, global_scores)

    def _popularity_weight_batch(
        self,
        batch: Tuple,
        item_popularity: Optional[torch.Tensor]
    ) -> Tuple:
        """
        Reweight interactions based on item popularity.
        Simulates popularity bias shift between domains.
        """
        user, pos_item, neg_item, context_items, context_scores, \
            global_items, global_scores = batch

        if context_scores is not None and item_popularity is not None:
            # Adjust scores based on item popularity
            pop_weights = item_popularity[context_items.clamp(min=0)]
            # Inverse popularity weighting (give more weight to unpopular items)
            inv_pop = 1.0 / (pop_weights + 1e-8)
            inv_pop = inv_pop / inv_pop.max()

            # Blend original scores with popularity-adjusted scores
            perturbed_scores = (
                (1 - self.popularity_alpha) * context_scores +
                self.popularity_alpha * inv_pop * context_scores
            )
        else:
            perturbed_scores = context_scores

        return (user, pos_item, neg_item, context_items, perturbed_scores,
                global_items, global_scores)

    def _noise_injection_batch(self, batch: Tuple) -> Tuple:
        """
        Add Gaussian noise to context scores.
        Simulates feature uncertainty in perturbed domain.
        """
        user, pos_item, neg_item, context_items, context_scores, \
            global_items, global_scores = batch

        if context_scores is not None:
            noise = torch.randn_like(context_scores) * self.noise_std
            perturbed_scores = context_scores + noise
            # Ensure non-negative scores
            perturbed_scores = torch.clamp(perturbed_scores, min=0)
        else:
            perturbed_scores = context_scores

        return (user, pos_item, neg_item, context_items, perturbed_scores,
                global_items, global_scores)

    def _interaction_mask_batch(self, batch: Tuple) -> Tuple:
        """
        Mask subset of interactions.
        Simulates incomplete interaction data.
        """
        user, pos_item, neg_item, context_items, context_scores, \
            global_items, global_scores = batch

        if context_items is not None:
            # Random masking with padding token
            mask = torch.rand_like(context_items.float()) > self.edge_dropout_rate
            mask[:, 0] = True  # Keep at least one

            # Replace masked items with padding token (0)
            perturbed_context = context_items.clone()
            perturbed_context[~mask] = 0

            perturbed_scores = context_scores.clone()
            perturbed_scores[~mask] = -100  # Padding score
        else:
            perturbed_context = context_items
            perturbed_scores = context_scores

        return (user, pos_item, neg_item, perturbed_context, perturbed_scores,
                global_items, global_scores)

    def _combined_perturbation_batch(
        self,
        batch: Tuple,
        item_popularity: Optional[torch.Tensor]
    ) -> Tuple:
        """
        Apply multiple perturbation strategies sequentially.
        """
        # First apply edge dropout
        batch = self._edge_dropout_batch(batch)
        # Then add noise
        batch = self._noise_injection_batch(batch)
        # Finally apply popularity weighting if available
        if item_popularity is not None:
            batch = self._popularity_weight_batch(batch, item_popularity)

        return batch


class GraphPerturbation:
    """
    Perturbation strategies at the graph level.
    Used for EASE matrix and adjacency matrix perturbation.
    """

    def __init__(
        self,
        dropout_rate: float = 0.2,
        noise_scale: float = 0.1
    ):
        self.dropout_rate = dropout_rate
        self.noise_scale = noise_scale

    def perturb_adjacency_matrix(
        self,
        adj_matrix: sp.csr_matrix,
        method: str = 'dropout'
    ) -> sp.csr_matrix:
        """
        Perturb adjacency matrix.

        Args:
            adj_matrix: Sparse adjacency matrix
            method: 'dropout' or 'noise'

        Returns:
            Perturbed adjacency matrix
        """
        if method == 'dropout':
            return self._edge_dropout_sparse(adj_matrix)
        elif method == 'noise':
            return self._add_noise_sparse(adj_matrix)
        else:
            return adj_matrix

    def _edge_dropout_sparse(
        self,
        adj_matrix: sp.csr_matrix
    ) -> sp.csr_matrix:
        """Apply edge dropout to sparse matrix."""
        # Convert to COO for easier manipulation
        coo = adj_matrix.tocoo()
        n_edges = coo.nnz

        # Randomly select edges to keep
        keep_mask = np.random.random(n_edges) > self.dropout_rate

        # Create new sparse matrix with remaining edges
        new_data = coo.data[keep_mask]
        new_row = coo.row[keep_mask]
        new_col = coo.col[keep_mask]

        perturbed = sp.coo_matrix(
            (new_data, (new_row, new_col)),
            shape=adj_matrix.shape
        )

        # Re-normalize
        return self._normalize_sparse(perturbed.tocsr())

    def _add_noise_sparse(
        self,
        adj_matrix: sp.csr_matrix
    ) -> sp.csr_matrix:
        """Add Gaussian noise to edge weights."""
        coo = adj_matrix.tocoo()

        # Add noise to existing edge weights
        noise = np.random.normal(0, self.noise_scale, coo.nnz)
        new_data = np.maximum(coo.data + noise, 0)

        perturbed = sp.coo_matrix(
            (new_data, (coo.row, coo.col)),
            shape=adj_matrix.shape
        )

        return self._normalize_sparse(perturbed.tocsr())

    def _normalize_sparse(self, adj_matrix: sp.csr_matrix) -> sp.csr_matrix:
        """Row-normalize sparse matrix."""
        rowsum = np.array(adj_matrix.sum(axis=1)).flatten()
        d_inv = np.power(rowsum + 1e-8, -1)
        d_mat = sp.diags(d_inv)
        return d_mat.dot(adj_matrix)

    def perturb_ease_matrix(
        self,
        ease_matrix: torch.Tensor,
        method: str = 'dropout'
    ) -> torch.Tensor:
        """
        Perturb EASE similarity matrix.

        Args:
            ease_matrix: Item-item similarity matrix (sparse or dense)
            method: Perturbation method

        Returns:
            Perturbed EASE matrix
        """
        if method == 'dropout':
            # Randomly zero out some similarities
            mask = torch.rand_like(ease_matrix) > self.dropout_rate
            return ease_matrix * mask.float()
        elif method == 'noise':
            # Add Gaussian noise
            noise = torch.randn_like(ease_matrix) * self.noise_scale
            perturbed = ease_matrix + noise
            return torch.clamp(perturbed, min=0)
        elif method == 'threshold':
            # Increase sparsity threshold
            threshold = ease_matrix.max() * 0.15  # Higher than original 0.1
            return ease_matrix * (ease_matrix > threshold).float()
        else:
            return ease_matrix


class EmbeddingPerturbation:
    """
    Perturb embeddings directly (feature-level perturbation).
    Simulates domain shift at feature statistics level.
    """

    def __init__(self, noise_std: float = 0.1):
        self.noise_std = noise_std

    def perturb_embeddings(
        self,
        embeddings: torch.Tensor,
        method: str = 'feature_stats'
    ) -> Tuple[torch.Tensor, float]:
        """
        Apply perturbation to embeddings.

        Args:
            embeddings: Embedding tensor [batch, dim] or [n_items, dim]
            method: 'feature_stats' or 'gaussian'

        Returns:
            (perturbed_embeddings, perturbation_magnitude)
        """
        if method == 'feature_stats':
            return self._feature_stats_perturbation(embeddings)
        elif method == 'gaussian':
            return self._gaussian_perturbation(embeddings)
        else:
            return embeddings, 0.0

    def _feature_stats_perturbation(
        self,
        embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, float]:
        """
        Perturb feature statistics (mean and variance).

        x' = β(x) * (x - μ(x)) / σ(x) + α(x)

        Where α and β are perturbed mean and std.
        """
        # Compute original statistics
        mu = embeddings.mean(dim=0, keepdim=True)
        sigma = embeddings.std(dim=0, keepdim=True) + 1e-8

        # Perturb statistics
        eps_mu = torch.randn_like(mu) * self.noise_std
        eps_sigma = torch.randn_like(sigma) * self.noise_std

        # New statistics
        alpha = mu + eps_mu
        beta = sigma * (1 + eps_sigma)

        # Apply perturbation
        normalized = (embeddings - mu) / sigma
        perturbed = beta * normalized + alpha

        # Compute perturbation magnitude (for DR-FIM weighting)
        perturbation_mag = (eps_mu.abs().mean() + eps_sigma.abs().mean()).item()

        return perturbed, perturbation_mag

    def _gaussian_perturbation(
        self,
        embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, float]:
        """Simple Gaussian noise injection."""
        noise = torch.randn_like(embeddings) * self.noise_std
        perturbed = embeddings + noise
        perturbation_mag = noise.abs().mean().item()

        return perturbed, perturbation_mag


class PopularityComputer:
    """
    Compute item popularity statistics for popularity-based perturbation.
    """

    @staticmethod
    def compute_from_interactions(
        interactions: Dict[int, List],
        num_items: int
    ) -> torch.Tensor:
        """
        Compute item popularity from interaction data.

        Args:
            interactions: Dictionary {user_id: [(item_id, score), ...]}
            num_items: Total number of items

        Returns:
            Popularity scores for each item
        """
        popularity = torch.zeros(num_items)

        for user_id, items in interactions.items():
            for item_info in items:
                if isinstance(item_info, (list, tuple)):
                    item_id = item_info[0]
                else:
                    item_id = item_info

                if 0 <= item_id < num_items:
                    popularity[item_id] += 1

        # Normalize to [0, 1]
        if popularity.max() > 0:
            popularity = popularity / popularity.max()

        return popularity

    @staticmethod
    def compute_from_edge_list(
        edge_list: List[Tuple[int, int]],
        num_items: int
    ) -> torch.Tensor:
        """
        Compute item popularity from edge list.

        Args:
            edge_list: List of (user_id, item_id) tuples
            num_items: Total number of items

        Returns:
            Popularity scores
        """
        popularity = torch.zeros(num_items)

        for user_id, item_id in edge_list:
            if 0 <= item_id < num_items:
                popularity[item_id] += 1

        if popularity.max() > 0:
            popularity = popularity / popularity.max()

        return popularity
