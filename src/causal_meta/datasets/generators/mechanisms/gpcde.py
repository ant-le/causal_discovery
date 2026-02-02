from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
from torch import nn

class GPMechanism(nn.Module):
    """
    Gaussian Process Mechanism approximated via Random Fourier Features (RFF).
    f(parents, noise) ~ GP(0, k([parents, noise], [parents', noise']))
    """
    def __init__(
        self,
        input_dim: int,
        rff_dim: int = 256,
        length_scale: float = 1.0,
        output_variance: float = 1.0,
        *,
        weights: torch.Tensor | None = None,
        biases: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        # Input dim = num_parents + 1 (for noise)
        self.input_dim = input_dim
        self.rff_dim = rff_dim
        self.length_scale = length_scale
        self.output_variance = output_variance
        
        # Sample RFF parameters (fixed for this mechanism instance)
        # Weights w ~ N(0, 1/l^2 I)
        # Biases b ~ U(0, 2pi)
        
        # Deterministic parameters should be provided by the factory via `torch_generator`.
        # If they are not provided, fall back to global RNG (discouraged for reproducibility).
        if weights is None:
            weights = torch.randn(input_dim, rff_dim) / length_scale
        if biases is None:
            biases = torch.rand(rff_dim) * 2 * torch.pi

        self.register_buffer("weights", weights)
        self.register_buffer("biases", biases)
        
        # Scale factor for RFF: sqrt(2/D) * sqrt(variance)
        self.scale = np.sqrt(2.0 / rff_dim) * np.sqrt(output_variance)

    def forward(self, parents: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # Concatenate parents and noise: (Batch, N_parents + 1)
        # noise is (Batch, 1) or (Batch,)
        noise = noise.view(-1, 1)
        if parents.numel() > 0:
            inp = torch.cat([parents, noise], dim=1)
        else:
            inp = noise
            
        # Compute RFF
        # Z = cos(XW + b)
        proj = inp @ self.weights + self.biases
        features = torch.cos(proj)
        
        # Output is sum of features (scaled)
        # We assume coefficients are all 1.0 (approximating GP sample) 
        # Actually RFF approximation of f(x) is theta^T phi(x) where theta ~ N(0, I).
        # But commonly we just sum them if we fold theta into the weights/phases or just sum cosine features.
        # Standard: f(x) = sqrt(2/M) * sum_{j=1}^M cos(w_j x + b_j). 
        # This approximates f ~ GP(0, k).
        
        output = features.sum(dim=1) * self.scale
        return output

class GPMechanismFactory:
    def __init__(self, rff_dim: int = 256, length_scale_range: tuple[float, float] = (0.5, 2.0), 
                 variance: float = 1.0) -> None:
        self.rff_dim = rff_dim
        self.length_scale_range = length_scale_range
        self.variance = variance

    def make_mechanism(self, input_dim: int, torch_generator: torch.Generator) -> nn.Module:
        # Sample length scale
        l_min, l_max = self.length_scale_range
        # Uniform sampling
        rand_val = torch.rand(1, generator=torch_generator).item()
        length_scale = l_min + rand_val * (l_max - l_min)
        
        # Input dim includes noise. Sample RFF parameters deterministically from torch_generator.
        total_input_dim = input_dim + 1
        weights = torch.randn(total_input_dim, self.rff_dim, generator=torch_generator) / length_scale
        biases = torch.rand(self.rff_dim, generator=torch_generator) * 2 * torch.pi

        return GPMechanism(
            total_input_dim,
            self.rff_dim,
            length_scale,
            self.variance,
            weights=weights,
            biases=biases,
        )

    def __call__(self, adjacency_matrix: torch.Tensor, *, torch_generator: Optional[torch.Generator] = None, rng: Optional[np.random.Generator] = None) -> List[nn.Module]:
        torch_generator = torch_generator or torch.Generator()
        mechanisms = []
        n_nodes = adjacency_matrix.shape[0]
        for node in range(n_nodes):
            parents = torch.nonzero(adjacency_matrix[:, node], as_tuple=False).flatten()
            mechanisms.append(self.make_mechanism(int(parents.numel()), torch_generator))
        return mechanisms
