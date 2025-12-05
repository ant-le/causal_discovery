from __future__ import annotations
from typing import Dict, Iterable, List, TYPE_CHECKING
import numpy as np
import torch
from scipy import stats

if TYPE_CHECKING:
    from causal_meta.datasets.scm import SCMFamily

def degree_sequence(adjacency: torch.Tensor) -> torch.Tensor:
    """Return total degree (in + out) for each node."""
    adj = adjacency.float()
    return adj.sum(dim=0) + adj.sum(dim=1)

def spectral_radius(adjacency: torch.Tensor) -> float:
    """Compute spectral radius using a symmetrized adjacency."""
    sym_adj = (adjacency + adjacency.T) / 2.0
    eigenvalues = torch.linalg.eigvalsh(sym_adj)
    return float(eigenvalues.abs().max().item())

def degree_histogram(degrees: Iterable[float]) -> np.ndarray:
    values = list(degrees)
    if not values:
        return np.array([1.0], dtype=float)
    max_degree = max(1, int(max(values)))
    hist, _ = np.histogram(values, bins=np.arange(0, max_degree + 2), density=True)
    hist = np.clip(hist, 1e-12, None)
    return hist / hist.sum()

def get_family_stats(family: SCMFamily, n_samples: int = 100) -> Dict[str, float]:
    """Compute average graph statistics for a family."""
    if n_samples < 1:
        raise ValueError("n_samples must be positive.")

    avg_degrees: List[float] = []
    sparsities: List[float] = []
    spectral_radii: List[float] = []

    for seed in range(n_samples):
        instance = family.sample_task(seed)
        adjacency = instance.adjacency_matrix.detach().cpu()
        degrees = degree_sequence(adjacency)
        avg_degrees.append(float(degrees.mean().item()))

        n_nodes = adjacency.shape[0]
        possible_edges = n_nodes * (n_nodes - 1) / 2
        sparsity = float(adjacency.sum().item() / possible_edges) if possible_edges > 0 else 0.0
        sparsities.append(sparsity)

        spectral_radii.append(spectral_radius(adjacency))

    return {
        "avg_degree": float(np.mean(avg_degrees)) if avg_degrees else 0.0,
        "sparsity": float(np.mean(sparsities)) if sparsities else 0.0,
        "spectral_radius": float(np.mean(spectral_radii)) if spectral_radii else 0.0,
    }

def compute_family_distance(
    family_a: SCMFamily, family_b: SCMFamily, metric: str = "spectral", n_samples: int = 50
) -> float:
    """Compute a distance between two SCM families."""
    if n_samples < 1:
        raise ValueError("n_samples must be positive.")

    metric = metric.lower()
    if metric == "spectral":
        stats_a = get_family_stats(family_a, n_samples=n_samples)
        stats_b = get_family_stats(family_b, n_samples=n_samples)
        return abs(stats_a["spectral_radius"] - stats_b["spectral_radius"])
    if metric == "kl":
        degrees_a: List[float] = []
        degrees_b: List[float] = []
        for seed in range(n_samples):
            adj_a = family_a.sample_task(seed).adjacency_matrix.detach().cpu()
            adj_b = family_b.sample_task(seed).adjacency_matrix.detach().cpu()
            degrees_a.extend(degree_sequence(adj_a).tolist())
            degrees_b.extend(degree_sequence(adj_b).tolist())

        p = degree_histogram(degrees_a)
        q = degree_histogram(degrees_b)
        # Align histogram lengths
        if p.shape[0] < q.shape[0]:
            p = np.pad(p, (0, q.shape[0] - p.shape[0]), constant_values=1e-12)
        elif q.shape[0] < p.shape[0]:
            q = np.pad(q, (0, p.shape[0] - q.shape[0]), constant_values=1e-12)
        return float(stats.entropy(p, q))

    raise ValueError(f"Unsupported metric: {metric}")
