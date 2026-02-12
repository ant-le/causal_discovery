from __future__ import annotations

from typing import List, Optional, Protocol

import numpy as np
import torch
from torch import nn


class MechanismFactory(Protocol):
    """Protocol for mechanism factories."""

    def make_mechanism(self, input_dim: int, rng: torch.Generator) -> nn.Module: ...

    def __call__(
        self,
        adjacency_matrix: torch.Tensor,
        *,
        torch_generator: Optional[torch.Generator] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> List[nn.Module]: ...

    def make_mechanism(
        self, input_dim: int, torch_generator: torch.Generator
    ) -> nn.Module: ...
