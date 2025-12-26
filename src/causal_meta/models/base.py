from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn


class BaseModel(nn.Module, ABC):
    """
    Abstract base class for all causal discovery models in the meta-learning framework.
    """

    def __init__(self) -> None:
        super().__init__()

    @property
    @abstractmethod
    def needs_pretraining(self) -> bool:
        """
        Indicates if the model requires a meta-training phase.
        
        Returns:
            True if the model is an amortized meta-learner (e.g., Avici).
            False if the model optimizes/samples per instance (e.g., MCMC, VI).
        """
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor) -> Any:
        """
        Forward pass of the model.
        
        Args:
            x: Input data tensor of shape (Batch, Samples, Variables).
            
        Returns:
            Model-specific output (e.g., logits, distribution parameters).
        """
        pass

    @abstractmethod
    def sample(self, x: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        """
        Generate causal structure samples.
        
        Args:
            x: Input data tensor of shape (Batch, Samples, Variables).
            n_samples: Number of graph samples to return per batch element.
            
        Returns:
            Predicted adjacency matrices of shape (Batch, n_samples, Variables, Variables).
        """
        pass

    @abstractmethod
    def calculate_loss(self, output: Any, target: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Calculate model-specific loss.

        Args:
            output: Model output from forward().
            target: Ground truth adjacency matrix.
            **kwargs: Additional arguments for loss calculation (e.g. update_regulariser).

        Returns:
            Loss tensor of shape (Batch,).
        """
        pass