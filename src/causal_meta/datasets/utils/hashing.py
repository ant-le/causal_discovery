from __future__ import annotations
import hashlib
import numpy as np
import torch

def compute_graph_hash(adj: torch.Tensor) -> str:
    """Return a deterministic hash for an adjacency matrix."""
    adj_bytes = adj.detach().cpu().numpy().astype(np.int8).tobytes()
    return hashlib.sha256(adj_bytes).hexdigest()
