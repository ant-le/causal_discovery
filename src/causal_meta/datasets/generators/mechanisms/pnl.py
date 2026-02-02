from __future__ import annotations
from typing import List, Optional, Callable, Any
import torch
from torch import nn
import numpy as np
from .linear import LinearMechanismFactory

class PNLMechanism(nn.Module):
    """Post-Nonlinear Mechanism: g(f(parents) + noise)."""
    def __init__(self, inner_mechanism: nn.Module, nonlinearity: Callable[[torch.Tensor], torch.Tensor]) -> None:
        super().__init__()
        self.inner_mechanism = inner_mechanism
        self.nonlinearity = nonlinearity

    def forward(self, parents: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # Compute inner mechanism output: Y = f(parents) + noise
        y = self.inner_mechanism(parents, noise)
        # Apply nonlinearity: X = g(Y)
        return self.nonlinearity(y)

class PNLMechanismFactory:
    """Factory for Post-Nonlinear mechanisms."""
    def __init__(self, inner_factory: Optional[Any] = None, nonlinearity_type: str = "cube") -> None:
        self.inner_factory = inner_factory or LinearMechanismFactory()
        self.nonlinearity_type = nonlinearity_type

    def _get_nonlinearity(self) -> Callable[[torch.Tensor], torch.Tensor]:
        if self.nonlinearity_type == "cube":
            return lambda x: x**3
        elif self.nonlinearity_type == "sigmoid":
            return torch.sigmoid
        elif self.nonlinearity_type == "tanh":
            return torch.tanh
        else:
            raise ValueError(f"Unknown nonlinearity type: {self.nonlinearity_type}")

    def __call__(
        self,
        adjacency_matrix: torch.Tensor,
        *,
        torch_generator: Optional[torch.Generator] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> List[nn.Module]:
        # Generate inner mechanisms
        inner_mechanisms = self.inner_factory(adjacency_matrix, torch_generator=torch_generator, rng=rng)
        
        # Wrap with PNL
        mechanisms: List[nn.Module] = []
        nonlinearity = self._get_nonlinearity()
        
        for inner in inner_mechanisms:
            mechanisms.append(PNLMechanism(inner, nonlinearity))
            
        return mechanisms
