from __future__ import annotations

import torch


def normalize_scm_data(x: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    """
    Standardize SCM observational data per feature.

    Supports:
      - (S, N): one task with S samples and N variables
      - (B, S, N): a batch of tasks

    Statistics are computed across all samples (and batch, if present), matching
    `collate_fn_scm` behavior.
    """
    if x.ndim not in {2, 3}:
        raise ValueError("Expected x to have shape (S, N) or (B, S, N).")

    flat = x.reshape(-1, x.shape[-1])
    mean = flat.mean(dim=0, keepdim=True)
    std = flat.std(dim=0, unbiased=False, keepdim=True).clamp_min(eps)
    return (x - mean) / std

