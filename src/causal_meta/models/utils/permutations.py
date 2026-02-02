import torch


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


def _greedy_row_matching(soft_perm: torch.Tensor) -> torch.Tensor:
    """
    Convert soft (approximately doubly-stochastic) matrices into hard permutation matrices.

    This is a fast, deterministic approximation (not Hungarian-optimal). It guarantees a
    valid permutation by selecting, for each row, the best currently-unused column.

    Args:
        soft_perm: (B, N, N)
    Returns:
        perm_hard: (B, N, N) with one 1 per row and per column.
    """
    if soft_perm.ndim != 3 or soft_perm.shape[-1] != soft_perm.shape[-2]:
        raise ValueError("soft_perm must be of shape (B, N, N).")

    bsz, n, _ = soft_perm.shape
    perm_hard = torch.zeros_like(soft_perm)
    neg_inf = torch.finfo(soft_perm.dtype).min

    for b in range(int(bsz)):
        used_cols = torch.zeros(n, dtype=torch.bool, device=soft_perm.device)
        for r in range(n):
            row = soft_perm[b, r]
            col = row.masked_fill(used_cols, neg_inf).argmax(dim=-1)
            perm_hard[b, r, col] = 1.0
            used_cols[col] = True

    return perm_hard


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
        hard: If True, returns hard permutation matrices (greedy assignment).
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
        perm_hard = _greedy_row_matching(soft_perm_flat.detach())
        soft_perm_flat = perm_hard - soft_perm_flat.detach() + soft_perm_flat

    perm = soft_perm_flat.reshape(B, n_samples, N, N)
    
    if squeeze and n_samples == 1:
        perm = perm.squeeze(1)
        
    return perm, soft_perm_flat
