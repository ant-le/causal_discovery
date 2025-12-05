from __future__ import annotations
from typing import Optional
import numpy as np
import torch

class ErdosRenyiGenerator:
    """Vectorized Erdős-Rényi DAG sampler using torch.bernoulli."""

    def __init__(self, edge_prob: float) -> None:
        if edge_prob < 0 or edge_prob > 1:
            raise ValueError("edge_prob must be in [0, 1].")
        self.edge_prob = edge_prob

    def __call__(
        self,
        n_nodes: int,
        *,
        seed: Optional[int] = None,
        torch_generator: Optional[torch.Generator] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> torch.Tensor:
        torch_generator = torch_generator or torch.Generator()
        if seed is not None:
            torch_generator = torch_generator.manual_seed(seed)

        mask = torch.triu(torch.ones((n_nodes, n_nodes)), diagonal=1)
        prob_matrix = mask * self.edge_prob
        adjacency = torch.bernoulli(prob_matrix, generator=torch_generator)
        return adjacency.float()
