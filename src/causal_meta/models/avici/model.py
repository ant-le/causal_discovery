from typing import Any, Optional

import torch
import torch.nn as nn
import torch.distributed as dist

from causal_meta.models.base import BaseModel
from causal_meta.models.factory import register_model
from causal_meta.models.utils.nn import CausalTNPEncoder, CausalAdjacencyMatrix

def cyclicity(adjacency: torch.Tensor) -> torch.Tensor:
    """
    Code adapted from DiBS:
    Differentiable acyclicity constraint from Yu et al. (2019). If h = 0 then the graph is acyclic.
    http://proceedings.mlr.press/v97/yu19a/yu19a.pdf

    Args:
        mat (ndarray): graph adjacency matrix of shape ``[n_vars, n_vars]``
        n_vars (int): number of variables, to allow for ``jax.jit``-compilation

    Returns:
        bool: True if the graph is cyclic, False otherwise
    """
    n_vars = adjacency.shape[-1]

    M_mult = torch.linalg.matrix_exp(adjacency)
    h = torch.einsum('...ii', M_mult) - n_vars

    return h


@register_model("avici")
class AviciModel(CausalTNPEncoder, BaseModel):
    """
    Avici-style Amortized Causal Discovery Model.
    Replicates the AviciDecoder architecture from the reference.
    """

    def __init__(
        self,
        num_nodes: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2, # num_layers_encoder
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        emb_depth: int = 1,
        use_positional_encoding: bool = False,
        **kwargs
    ) -> None:
        CausalTNPEncoder.__init__(
            self,
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            nhead=nhead,
            num_layers=num_layers,
            emb_depth=emb_depth,
            use_positional_encoding=use_positional_encoding,
            num_nodes=num_nodes,
            dropout=dropout,
            device=None,
            dtype=None,
            avici_summary=True # Max pooling
        )
        self.d_model = d_model
        
        # Decoder is a linear layer (Identity in reference)
        self.decoder = nn.Identity()

        # Predictor: Attention-based adjacency matrix
        self.predictor = CausalAdjacencyMatrix(
            nhead=1, # Single head for final prediction
            d_model=d_model,
            device=None,
            dtype=None,
        )

        self.regulariser_weight = torch.nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.regulariser_lr = 1e-4 # hard coded from avici paper
        self.cyclicity_value_avg = None

    @property
    def needs_pretraining(self) -> bool:
        return True

    def decode(self, representation: torch.Tensor) -> torch.Tensor:
        return self.decoder(representation)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Avici model.

        Args:
            input_data: Input tensor of shape (Batch, Samples, Variables).

        Returns:
            logits: Predicted adjacency logits of shape (Batch, Variables, Variables).
        """
        # input_data is [Batch, Samples, Variables] -> [Batch, Samples, Variables, 1]
        target_data = input_data.unsqueeze(-1)
        
        # Encode
        # representation: [Batch, Nodes, 1, d_model]
        representation = self.encode(target_data=target_data, mask=None)
        
        # Decode
        representation = representation.squeeze(2) # [Batch, Nodes, d_model]
        decoded_output = self.decode(representation)
        
        # Predict Adjacency
        # adj_matrix: [Batch, Nodes, Nodes]
        adj_matrix = self.predictor(decoded_output, padding_mask=None)
        
        return adj_matrix

    def sample(self, input_data: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """
        Bernoulli sampling from edge probabilities.
        
        Args:
            input_data: Input tensor of shape (Batch, Samples, Variables).
            num_samples: Number of graph samples to generate.
            
        Returns:
            torch.Tensor: Sampled adjacency matrices (Batch, num_samples, Variables, Variables).
        """
        logits = self.forward(input_data)
        probs = torch.sigmoid(logits)
        probs = probs * (1 - torch.eye(probs.size(-1), device=probs.device, dtype=probs.dtype))
        
        # Expand for num_samples
        probs = probs.unsqueeze(1).expand(-1, num_samples, -1, -1)
        
        # Sample
        return torch.bernoulli(probs)

    def update_regulariser_weight(self, acyclic_loss):
        """
        Should update every 250 steps.
        """
        self.regulariser_weight.data = self.regulariser_weight.data + self.regulariser_lr * acyclic_loss

    def calculate_loss(self, logits, target, update_regulariser=False, **kwargs):
        """
        Args:
        -----
            logits: torch.Tensor, shape [batch_size, num_samples, num_nodes, num_nodes]
            target: torch.Tensor, shape [batch_size, num_nodes, num_nodes]

        Returns:
        --------
            loss: torch.Tensor, shape [batch_size]
            logits: torch.Tensor, shape [batch_size, num_nodes ** 2]
        """
        probs = torch.sigmoid(logits)
        # set diagonal to 0
        probs = probs * (1 - torch.eye(probs.size(-1), device=probs.device))
        
        # Differentiable acyclicity constraint (per batch element)
        # NOTE: keep this as a tensor to preserve gradients.
        cyclicity_per_example = cyclicity(probs)  # (B,)
        cyclicity_mean = cyclicity_per_example.mean()

        # Sync a detached scalar for logging/EMA so all ranks update the dual weight identically.
        cyclicity_logged = cyclicity_mean.detach()
        if dist.is_available() and dist.is_initialized():
            cyclicity_logged = cyclicity_logged.clone()
            dist.all_reduce(cyclicity_logged, op=dist.ReduceOp.AVG)

        cyclicity_value = float(cyclicity_logged.item())

        logits = logits.contiguous().view(logits.size(0), -1)
        target = target.contiguous().view(target.size(0), -1)
        # Classification loss
        loss_func = torch.nn.BCEWithLogitsLoss(reduction="none")
        loss = loss_func(logits, target)
        loss = loss.mean(dim=1)

        # Update EMA of cyclicity_value
        alpha = 0.1  # Smoothing factor between 0 and 1
        if self.cyclicity_value_avg is None:
            self.cyclicity_value_avg = cyclicity_value
        else:
            self.cyclicity_value_avg = (
                alpha * cyclicity_value + (1 - alpha) * self.cyclicity_value_avg
            )

        acyclic_loss = self.regulariser_weight * cyclicity_per_example

        # Update dual weight with EMA
        if update_regulariser:
            self.update_regulariser_weight(self.cyclicity_value_avg)
            
        total_loss = loss + acyclic_loss
        return total_loss.mean()
