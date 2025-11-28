import torch
import numpy as np
from typing import List, Callable, Optional

class LinearMechanism:
    """
    Linear Additive Noise Mechanism: X_j = sum(W_ij * X_i) + Noise
    """
    def __init__(self, num_parents: int, weights: Optional[np.ndarray] = None):
        self.num_parents = num_parents
        if weights is not None:
            self.weights = torch.tensor(weights, dtype=torch.float32)
        else:
            # Initialize random weights if not provided
            self.weights = torch.randn(num_parents, dtype=torch.float32)

    def __call__(self, parents_values: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Args:
            parents_values: Tensor of shape (batch, num_parents)
            noise: Tensor of shape (batch,)
        Returns:
            Tensor of shape (batch,)
        """
        if self.num_parents == 0:
            return noise
            
        # Linear combination: (batch, parents) @ (parents,) -> (batch,)
        effect = torch.matmul(parents_values, self.weights)
        return effect + noise