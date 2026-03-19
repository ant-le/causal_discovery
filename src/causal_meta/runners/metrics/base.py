from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
import torch
import torch.distributed as dist

from causal_meta.datasets.scm import SCMFamily
from causal_meta.datasets.utils.normalization import compute_scm_stats

log = logging.getLogger(__name__)


class BaseMetrics:
    """
    Shared base for metric trackers that:
    - accumulate per-batch scalars into `history: Dict[str, List[float]]`
    - support distributed gather (all_gather_object) for history
    - support memory-efficient distributed aggregation via all_reduce (sum/count)
    - provide helper for on-the-fly interventional data generation
    """

    def __init__(self) -> None:
        self.history: Dict[str, List[float]] = defaultdict(list)

    def reset(self) -> None:
        self.history = defaultdict(list)

    def get_raw_results(self) -> Dict[str, List[float]]:
        return self._gather_history()

    @staticmethod
    def _generate_interventional_data(
        family: SCMFamily,
        seed: int,
        n_samples: int,
        intervention_value: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Helper to generate a standard set of single-node interventions for an SCM instance.
        Returns a list of dicts: {"target": int, "value": float, "data": normalized_tensor}.
        """
        instance = family.sample_task(seed)
        n_nodes = family.n_nodes

        # 1. Get Normalization Stats from observational data
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            x_obs_raw = instance.sample(n_samples)

        mean, std = compute_scm_stats(x_obs_raw)

        # 2. Generate Interventions
        interventional_data = []
        for target_node in range(n_nodes):
            mutilated = instance.do({target_node: intervention_value})
            # Derived seed for reproducibility (consistent with datasets)
            int_seed = seed + (target_node + 1) * 1000

            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int_seed)
                x_int_raw = mutilated.sample(n_samples)

            # Normalize using observational stats
            x_int_norm = (x_int_raw - mean) / std

            interventional_data.append(
                {"target": target_node, "value": intervention_value, "data": x_int_norm}
            )

        return interventional_data

    def _gather_history(self) -> Dict[str, List[float]]:
        return self.gather(dict(self.history))

    def gather(self, local_obj: Dict[str, List[float]]) -> Dict[str, List[float]]:
        """
        Generic gather helper for tests/utility code.

        Args:
            local_obj: dict of lists on the current rank.
        Returns:
            merged dict of lists across all ranks (or local_obj if not distributed).
        """
        if not (dist.is_available() and dist.is_initialized()):
            return dict(local_obj)

        world_size = dist.get_world_size()
        gathered: List[Dict[str, List[float]] | None] = [
            None for _ in range(world_size)
        ]
        dist.all_gather_object(gathered, local_obj)

        merged: Dict[str, List[float]] = defaultdict(list)
        for item in gathered:
            if not item:
                continue
            for k, v in item.items():
                merged[k].extend(v)
        return dict(merged)

    def _reduce_history_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Memory-efficient distributed aggregation via all_reduce.

        Instead of gathering all raw data to rank 0 (OOM risk on large datasets),
        this computes local sum, sum-of-squares, and count, then reduces across
        ranks to compute global mean/std/sem.

        Returns:
            Dict mapping metric keys to {"mean", "std", "sem", "count"} dicts.
        """
        if not (dist.is_available() and dist.is_initialized()):
            # Non-distributed: compute stats directly from local history
            results: Dict[str, Dict[str, float]] = {}
            for k, v in self.history.items():
                if not v:
                    continue
                arr = np.asarray(v, dtype=float)
                arr = arr[np.isfinite(arr)]
                if arr.size == 0:
                    continue
                results[k] = {
                    "mean": float(arr.mean()),
                    "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
                    "sem": float(arr.std(ddof=1) / np.sqrt(arr.size))
                    if arr.size > 1
                    else 0.0,
                    "count": float(arr.size),
                }
            return results

        # Distributed: use all_reduce for sum/sum_sq/count
        try:
            backend = dist.get_backend()
        except Exception:
            backend = "gloo"
        if backend == "nccl" and torch.cuda.is_available():
            device = torch.device(f"cuda:{torch.cuda.current_device()}")
        else:
            device = torch.device("cpu")

        # Collect all keys across ranks (may differ due to batch distribution)
        local_keys = set(self.history.keys())
        all_keys_list: List[set[str] | None] = [
            None for _ in range(dist.get_world_size())
        ]
        dist.all_gather_object(all_keys_list, local_keys)
        all_keys = set()
        for ks in all_keys_list:
            if ks:
                all_keys.update(ks)

        results = {}
        for k in all_keys:
            v = self.history.get(k, [])
            arr = np.asarray(v, dtype=float) if v else np.array([], dtype=float)
            arr = arr[np.isfinite(arr)]

            local_sum = float(arr.sum()) if arr.size > 0 else 0.0
            local_sum_sq = float((arr**2).sum()) if arr.size > 0 else 0.0
            local_count = float(arr.size)

            # Create tensor with [sum, sum_sq, count]
            stats = torch.tensor([local_sum, local_sum_sq, local_count], device=device)
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)

            total_sum, total_sum_sq, total_count = (
                float(stats[0].item()),
                float(stats[1].item()),
                float(stats[2].item()),
            )

            if total_count < 1:
                continue

            mean = total_sum / total_count
            # Variance via E[X^2] - E[X]^2 (population variance)
            variance = (total_sum_sq / total_count) - (mean**2)
            # Clamp numerical issues
            variance = max(0.0, variance)

            # Convert to sample std (Bessel's correction)
            if total_count > 1:
                std = np.sqrt(variance * total_count / (total_count - 1))
                sem = std / np.sqrt(total_count)
            else:
                std = 0.0
                sem = 0.0

            results[k] = {
                "mean": mean,
                "std": std,
                "sem": sem,
                "count": total_count,
            }

        return results

    def summarize_distributed(self, *, summary_stats: bool = True) -> Dict[str, Any]:
        """
        Summarize history with distributed aggregation.

        Uses memory-efficient all_reduce to compute global stats without
        gathering all raw data to a single rank.

        Args:
            summary_stats: If True, returns {metric}_mean/sem/std. Else returns mean only.

        Returns:
            Dictionary of summarized metrics.
        """
        # Memory-efficient: compute stats via all_reduce
        reduced = self._reduce_history_stats()
        results: Dict[str, Any] = {}
        for k, stats in reduced.items():
            if summary_stats:
                results[f"{k}_mean"] = stats["mean"]
                results[f"{k}_sem"] = stats["sem"]
                results[f"{k}_std"] = stats["std"]
            else:
                results[k] = stats["mean"]
        return results
