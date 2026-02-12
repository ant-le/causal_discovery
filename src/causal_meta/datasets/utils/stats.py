from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Iterable, List

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
        sparsity = (
            float(adjacency.sum().item() / possible_edges)
            if possible_edges > 0
            else 0.0
        )
        sparsities.append(sparsity)

        spectral_radii.append(spectral_radius(adjacency))

    return {
        "avg_degree": float(np.mean(avg_degrees)) if avg_degrees else 0.0,
        "sparsity": float(np.mean(sparsities)) if sparsities else 0.0,
        "spectral_radius": float(np.mean(spectral_radii)) if spectral_radii else 0.0,
    }


def compute_family_distance(
    family_a: SCMFamily,
    family_b: SCMFamily,
    metric: str = "spectral",
    n_samples: int = 50,
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


def sliced_wasserstein_distance(
    x: torch.Tensor, y: torch.Tensor, num_projections: int = 100
) -> float:
    """
    Compute the Sliced Wasserstein Distance (SWD) between two distributions.

    Approximates the Wasserstein distance by projecting the high-dimensional
    distributions onto random one-dimensional lines (slices). The distance is
    then computed as the average 2-Wasserstein distance between the 1D projections.

    Args:
        x: Samples from the first distribution, shape (N, D).
        y: Samples from the second distribution, shape (M, D).
        num_projections: Number of random projections to use.

    Returns:
        The estimated Sliced Wasserstein Distance.
    """
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("Input tensors must be 2D (samples, features).")
    if x.shape[1] != y.shape[1]:
        raise ValueError("Input tensors must have the same number of features.")

    dim = x.shape[1]

    # Generate random projections on the unit sphere
    projections = torch.randn((dim, num_projections), device=x.device, dtype=x.dtype)
    projections = projections / torch.linalg.norm(projections, dim=0, keepdim=True)

    # Project the data
    x_proj = x @ projections  # (N, num_projections)
    y_proj = y @ projections  # (M, num_projections)

    # Sort the projections
    x_proj_sorted, _ = torch.sort(x_proj, dim=0)
    y_proj_sorted, _ = torch.sort(y_proj, dim=0)

    # Note: This simple implementation assumes equal sample sizes (N=M).
    if x.shape[0] != y.shape[0]:
        raise ValueError("For simple SWD, sample sizes must be equal.")

    # Compute L2 distance between sorted projections (Wasserstein-2 on 1D)
    diff = x_proj_sorted - y_proj_sorted
    w2_squared = torch.mean(diff**2, dim=0)
    swd_squared = torch.mean(w2_squared)

    return float(torch.sqrt(swd_squared).item())


def compute_distribution_distance(
    family_a: SCMFamily,
    family_b: SCMFamily,
    metric: str = "wasserstein",
    n_samples: int = 50,
    samples_per_task: int = 1000,
    num_projections: int = 100,
) -> float:
    """
    Compute a distributional distance between two SCM families.

    Args:
        family_a: First SCM family.
        family_b: Second SCM family.
        metric: Distance metric to use. Currently only "wasserstein".
        n_samples: Number of tasks (seeds) to average over.
        samples_per_task: Number of data samples per task.
        num_projections: Number of projections for SWD.

    Returns:
        Average distance between matched tasks.
    """
    if n_samples < 1:
        raise ValueError("n_samples must be positive.")

    metric = metric.lower()
    if metric == "wasserstein":
        distances = []
        for seed in range(n_samples):
            task_a = family_a.sample_task(seed)
            task_b = family_b.sample_task(seed)

            x = task_a.sample(samples_per_task)
            y = task_b.sample(samples_per_task)

            # Ensure on same device (CPU is safest for simple stats)
            x = x.cpu()
            y = y.cpu()

            dist = sliced_wasserstein_distance(x, y, num_projections=num_projections)
            distances.append(dist)

        return float(np.mean(distances))

    raise ValueError(f"Unsupported metric: {metric}")
