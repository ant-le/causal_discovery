import torch
import torch.nn.functional as F


def sinkhorn(log_alpha: torch.Tensor, n_iters: int = 20) -> torch.Tensor:
    """
    Sinkhorn-Knopp algorithm for doubly stochastic matrices.
    Args:
        log_alpha: Log-probabilities (Batch, N, N).
        n_iters: Number of normalization iterations.
    Returns:
        Doubly stochastic matrix (Batch, N, N).
    """
    for _ in range(n_iters):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-1, keepdim=True)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-2, keepdim=True)
    return torch.exp(log_alpha)


def sample_permutation(
    log_alpha: torch.Tensor,
    temp: float = 1.0,
    noise_factor: float = 1.0,
    n_samples: int = 1,
    hard: bool = False,
    n_iters: int = 20,
    squeeze: bool = False,
    device: torch.device = None
) -> torch.Tensor:
    """
    Sample permutations using the Gumbel-Sinkhorn method.
    
    Args:
        log_alpha: Log-probabilities (Batch, N, N).
        temp: Temperature for Gumbel-Softmax.
        noise_factor: Scale of Gumbel noise.
        n_samples: Number of samples per batch element.
        hard: If True, returns hard permutation matrices (Hungarian alg approximation).
        n_iters: Number of Sinkhorn iterations.
        squeeze: If True and n_samples=1, squeeze the sample dimension.
        device: Device to create tensors on.
        
    Returns:
        Sampled permutations (Batch, n_samples, N, N) or (Batch, N, N).
    """
    B, N, _ = log_alpha.shape
    
    # Expand for samples: (B, S, N, N)
    log_alpha_ex = log_alpha.unsqueeze(1).expand(-1, n_samples, -1, -1)
    
    # Gumbel Noise
    noise = -torch.log(-torch.log(torch.rand_like(log_alpha_ex) + 1e-20) + 1e-20)
    log_alpha_w_noise = (log_alpha_ex + noise * noise_factor) / temp
    
    # Sinkhorn
    # Collapse batch and samples for processing: (B*S, N, N)
    log_alpha_flat = log_alpha_w_noise.reshape(B * n_samples, N, N)
    
    soft_perm_flat = sinkhorn(log_alpha_flat, n_iters=n_iters)
    
    if hard:
        try:
            from scipy.optimize import linear_sum_assignment
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("hard=True requires scipy (scipy.optimize.linear_sum_assignment).") from exc

        perm_hard = torch.zeros_like(soft_perm_flat)
        for idx, matrix in enumerate(soft_perm_flat.detach().cpu().numpy()):
            row_ind, col_ind = linear_sum_assignment(-matrix)
            perm_hard[idx, row_ind, col_ind] = 1.0
        perm_hard = perm_hard.to(device=soft_perm_flat.device, dtype=soft_perm_flat.dtype)
        soft_perm_flat = perm_hard - soft_perm_flat.detach() + soft_perm_flat

    perm = soft_perm_flat.reshape(B, n_samples, N, N)
    
    if squeeze and n_samples == 1:
        perm = perm.squeeze(1)
        
    return perm, soft_perm_flat
