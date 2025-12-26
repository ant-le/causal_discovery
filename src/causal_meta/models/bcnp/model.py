from typing import Any, Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from causal_meta.models.base import BaseModel
from causal_meta.models.factory import register_model
from causal_meta.models.utils.nn import CausalTNPEncoder, CausalAdjacencyMatrix, CausalTransformerDecoderLayer, build_mlp
from causal_meta.models.utils.permutations import sample_permutation


@register_model("bcnp")
class BCNP(CausalTNPEncoder, BaseModel):
    """
    Bayesian Causal Neural Process (BCNP).
    Replicates the CausalProbabilisticDecoder architecture from the reference.
    Generates DAGs via Sinkhorn Permutations: A = P @ L @ P.T
    """

    def __init__(
        self,
        num_nodes: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2, # encoder layers
        num_layers_decoder: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        emb_depth: int = 1,
        use_positional_encoding: bool = False,
        n_perm_samples: int = 10,
        sinkhorn_iter: int = 20,
        q_before_l: bool = True,
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
            avici_summary=False # BCNP uses Attention Summary (default)
        )
        self.d_model = d_model
        
        self.num_nodes = num_nodes
        self.n_perm_samples = n_perm_samples
        self.sinkhorn_iter = sinkhorn_iter
        self.q_before_l = q_before_l
        
        # Decoders
        # The reference splits num_layers_decoder into two halves
        decoder_layers_half = max(1, num_layers_decoder // 2)
        
        self.decoder_L = nn.TransformerDecoder(
            decoder_layer=CausalTransformerDecoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout,
                norm_first=True, batch_first=True, bias=False
            ),
            num_layers=decoder_layers_half
        )
        
        self.decoder_Q = nn.TransformerDecoder(
            decoder_layer=CausalTransformerDecoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout,
                norm_first=True, batch_first=True, bias=False
            ),
            num_layers=decoder_layers_half
        )
        
        self.Q_param = CausalAdjacencyMatrix(nhead=nhead, d_model=d_model, device=None, dtype=None)
        self.L_param = CausalAdjacencyMatrix(nhead=nhead, d_model=d_model, device=None, dtype=None)
        
        self.p_param = build_mlp(dim_in=d_model, dim_hid=d_model, dim_out=1, depth=emb_depth)

    @property
    def needs_pretraining(self) -> bool:
        return True

    def decode(self, representation: torch.Tensor, mask: Optional[torch.Tensor] = None):
        if not self.q_before_l:
            L_rep = self.decoder_L(representation, memory=None, tgt_key_padding_mask=mask)
            Q_rep = self.decoder_Q(L_rep, memory=None, tgt_key_padding_mask=mask)
        else:
            Q_rep = self.decoder_Q(representation, memory=None, tgt_key_padding_mask=mask)
            L_rep = self.decoder_L(Q_rep, memory=None, tgt_key_padding_mask=mask)
            
        # Get L matrix parameters
        L_param = self.L_param(L_rep, padding_mask=mask)
        # Symmetrize L_param for permutation equivariance (as per reference)
        L_param = (L_param + L_param.transpose(1, 2)) / 2
        
        return L_param, Q_rep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns edge probabilities marginalized over permutations.
        Shape: (n_perm_samples, Batch, V, V)
        Note: BaseModel usually expects (Batch, ...). 
        However, BCNP returns samples of prob matrices or a distribution.
        Here we follow the reference 'forward' which returns all_probs: (Samples, Batch, N, N).
        We might need to transpose to (Batch, Samples, N, N) to match BaseModel.sample?
        
        Wait, 'forward' in BaseModel returns "Model-specific output".
        The runner handles loss. The reference `calculate_loss` expects `probs`.
        So returning the (Samples, Batch, N, N) probs is correct for loss calculation.
        """
        B, S, V = x.shape
        if V != self.num_nodes:
            # In meta-learning, V might vary if model allows, but this implementation binds specific sizes 
            # for PositionalEncoding/masks. Reference throws error.
            raise ValueError("Number of nodes in the input data should be equal to num_nodes.")

        target_data = x.unsqueeze(-1)
        representation = self.encode(target_data=target_data, mask=None)
        representation = representation.squeeze(2) # (B, V, D)
        
        L_param, Q_rep = self.decode(representation, mask=None)
        
        # Calculate Q parameters (permutation logits)
        p_param = self.p_param(Q_rep).squeeze(-1) # (B, V)
        ovector = torch.arange(1, self.num_nodes + 1, device=x.device, dtype=x.dtype)
        
        # Outer product to get matrix logits
        Q_param = torch.einsum("bn,m->bnm", p_param, ovector[:V])
        Q_param = F.logsigmoid(Q_param)
        
        # Mask diagonal
        Q_mask = 1 - torch.eye(V, device=x.device)
        # The reference logic for Q_mask seems to apply to Q_param addition?
        # Reference: Q_mask = ...; Q_param = Q_param + Q_mask
        # If Q_mask is 0 on diagonal and 1 elsewhere? 
        # Reference: Q_mask = decoder_mask... triu/tril?
        # Actually reference: Q_mask = (1 - eye). This ensures diagonal is not suppressed? 
        # Wait, if Q_param is log_sigmoid (negative), adding 0 changes nothing. 
        # Reference logic is slightly opaque on masking without variable size. 
        # We will assume standard full permutation for now.
        
        # Sample Permutations
        perm, _ = sample_permutation(
            log_alpha=Q_param,
            temp=1.0,
            noise_factor=1.0,
            n_samples=self.n_perm_samples,
            hard=True,
            n_iters=self.sinkhorn_iter,
            squeeze=False,
            device=x.device
        )
        # perm: (B, K, N, N) -> Transpose to (B, K, N, N) (Reference returns (B, K, N, N) then transposes?)
        # Reference: perm = perm.transpose(1, 0) (Sample, Batch, ...) ??
        # Reference `sample_permutation` returns (B, K, N, N).
        # Reference forward: perm = perm.transpose(1, 0) -> (K, B, N, N).
        perm = perm.transpose(1, 0) # (K, B, N, N)
        perm_inv = perm.transpose(3, 2)
        
        # Lower Triangular Mask
        mask = torch.tril(torch.ones((V, V), device=x.device), diagonal=-1)
        
        # P @ Mask @ P.T
        # perm: (K, B, N, N)
        # mask: (N, N)
        # Einstein sum:
        # perm (K, B, i, j) * mask (j, k) * perm_inv (K, B, k, l) -> (K, B, i, l)
        all_masks = torch.einsum("nbij,jk,nbkl->nbil", perm, mask, perm_inv)
        
        # Probs from L
        probs = torch.sigmoid(L_param) # (B, N, N)
        
        # Elementwise mul
        all_probs = probs.unsqueeze(0) * all_masks # (K, B, N, N)
        
        return all_probs

    def sample(self, x: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        """
        Returns sampled graphs.
        Shape: (Batch, n_samples, N, N)
        """
        # We temporarily set n_perm_samples to requested n_samples to get K=n_samples
        original_k = self.n_perm_samples
        self.n_perm_samples = n_samples
        
        all_probs = self.forward(x) # (K, B, N, N)
        
        self.n_perm_samples = original_k
        
        # Sample Bernoulli
        samples = torch.bernoulli(all_probs)
        
        # Transpose to (B, K, N, N) to match BaseModel interface
        samples = samples.permute(1, 0, 2, 3)
        
        return samples


    def calculate_loss(self, probs, target, **kwargs):
        """
        Args:
        -----
            probs: torch.Tensor, shape [num_samples, batch_size, num_nodes, num_nodes]
            target: torch.Tensor, shape [batch_size, num_nodes, num_nodes]

        Returns:
        --------
            loss: torch.Tensor, shape [batch_size]
        """
        # Reshape the last axis
        probs = probs.contiguous().view(probs.size(0), probs.size(1), -1)
        target_graph = target.reshape(target.size(0), -1)
        # Calculate the loss
        existence_dist = torch.distributions.Bernoulli(
            probs=probs
        )
        log_prob = existence_dist.log_prob(target_graph[None])
        # # Mean across pemutation samples
        log_prob_sum = torch.logsumexp(log_prob, dim=0) - math.log(log_prob.size(0))
        # # shape [batch, num_nodes**2]
        loss_per_edge = - log_prob_sum
        loss = loss_per_edge.mean(dim=1)
        return loss.mean()

