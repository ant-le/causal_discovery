from __future__ import annotations
from typing import List, Optional
import numpy as np
import torch
from torch import nn

class MLPMechanism(nn.Module):
    """Two-layer MLP mechanism with noise concatenated to inputs."""

    def __init__(self, input_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, parents: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat([parents, noise], dim=1)
        return self.net(inputs).squeeze(-1)


class MLPMechanismFactory:
    """Factory producing two-layer MLP mechanisms."""

    def __init__(self, hidden_dim: int = 32) -> None:
        self.hidden_dim = hidden_dim

    def make_mechanism(self, input_dim: int, torch_generator: torch.Generator) -> nn.Module:
        # Initialize module on CPU (default). We want random weights to be deterministic
        # based on torch_generator. PyTorch init uses global RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(torch.randint(0, 1000000, (1,), generator=torch_generator).item())
            mech = MLPMechanism(input_dim=input_dim, hidden_dim=self.hidden_dim)
        return mech

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
            input_dim = int(parents.numel())
            mechanisms.append(self.make_mechanism(input_dim, torch_generator))
        return mechanisms