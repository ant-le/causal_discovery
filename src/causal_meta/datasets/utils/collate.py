from __future__ import annotations
from typing import Iterable, Tuple
import torch

def collate_fn_scm(batch: Iterable[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collate function for SCM batches that stacks and normalizes data."""
    data, adjs = zip(*batch)
    data_tensor = torch.stack([x.float() for x in data])
    adj_tensor = torch.stack([a.float() for a in adjs])

    flat = data_tensor.reshape(-1, data_tensor.shape[-1])
    mean = flat.mean(dim=0, keepdim=True)
    std = flat.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-8)
    normalized = (data_tensor - mean) / std

    return normalized, adj_tensor
