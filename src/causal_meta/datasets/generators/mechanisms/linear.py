from __future__ import annotations
from typing import List, Optional
import numpy as np
import torch
from torch import nn

class LinearMechanism(nn.Module):
    """Additive linear mechanism with Gaussian noise."""

    def __init__(self, weights: torch.Tensor, noise_scale: float = 1.0) -> None:
        super().__init__()
        self.register_buffer("weights", weights.view(-1))
        self.register_buffer("noise_scale", torch.tensor(float(noise_scale), dtype=torch.float32))

    def forward(self, parents: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        linear_term = parents @ self.weights if parents.numel() > 0 else torch.zeros(
            parents.shape[0], device=parents.device, dtype=parents.dtype
        )
        return linear_term + noise.view(-1) * self.noise_scale


class LinearMechanismFactory:
    """Factory producing linear mechanisms aligned with the adjacency."""

    def __init__(self, weight_scale: float = 1.0, noise_concentration: float = 2.0, noise_rate: float = 2.0) -> None:
        self.weight_scale = weight_scale
        self.noise_concentration = noise_concentration
        self.noise_rate = noise_rate

    def __call__(
        self,
        adjacency_matrix: torch.Tensor,
        *,
        torch_generator: Optional[torch.Generator] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> List[nn.Module]:
        torch_generator = torch_generator or torch.Generator()

        mechanisms: List[nn.Module] = []
        n_nodes = adjacency_matrix.shape[0]
        for node in range(n_nodes):
            parents = torch.nonzero(adjacency_matrix[:, node], as_tuple=False).flatten()
            n_parents = int(parents.numel())
            if n_parents > 0:
                weights = torch.randn(n_parents, generator=torch_generator) * self.weight_scale
            else:
                weights = torch.zeros(0)

            noise_scale = torch.distributions.Gamma(self.noise_concentration, self.noise_rate).sample()
            mechanisms.append(LinearMechanism(weights, noise_scale=noise_scale.item()))

        return mechanisms
