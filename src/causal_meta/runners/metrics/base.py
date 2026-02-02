from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, List, Mapping

import numpy as np
import torch
import torch.distributed as dist

from causal_meta.datasets.scm import SCMFamily

log = logging.getLogger(__name__)


class BaseMetrics:
    """
    Shared base for metric trackers that:
    - accumulate per-batch scalars into `history: Dict[str, List[float]]`
    - support distributed gather (all_gather_object) for history
    - support distributed sync (all_reduce) for scalar dicts
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

        mean = x_obs_raw.mean(dim=0, keepdim=True)
        std = x_obs_raw.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-8)

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
        if not (dist.is_available() and dist.is_initialized()):
            return dict(self.history)

        world_size = dist.get_world_size()
        gathered_results: List[Dict[str, List[float]] | None] = [
            None for _ in range(world_size)
        ]
        dist.all_gather_object(gathered_results, dict(self.history))

        merged: Dict[str, List[float]] = defaultdict(list)
        for res in gathered_results:
            if not res:
                continue
            for k, v in res.items():
                merged[k].extend(v)

        return dict(merged)

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

    def sync(self, metrics: Mapping[str, float]) -> Dict[str, float]:
        """
        Synchronize scalar metrics across ranks via all_reduce(AVG).
        """
        if not (dist.is_available() and dist.is_initialized()):
            return dict(metrics)

        try:
            backend = dist.get_backend()
        except Exception:
            backend = "gloo"
        if backend == "nccl" and torch.cuda.is_available():
            device = torch.device(f"cuda:{torch.cuda.current_device()}")
        else:
            device = torch.device("cpu")

        synced: Dict[str, float] = {}
        for k, v in metrics.items():
            t = torch.tensor(float(v), device=device)
            dist.all_reduce(t, op=dist.ReduceOp.AVG)
            synced[k] = float(t.detach().cpu().item())
        return synced

    @staticmethod
    def _summarize_history(
        history: Dict[str, List[float]], *, summary_stats: bool
    ) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        for k, v in history.items():
            if not v:
                continue
            arr = np.asarray(v, dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                continue

            if summary_stats:
                results[f"{k}_mean"] = float(arr.mean())
                if arr.size > 1:
                    results[f"{k}_sem"] = float(arr.std(ddof=1) / np.sqrt(arr.size))
                    results[f"{k}_std"] = float(arr.std(ddof=1))
                else:
                    results[f"{k}_sem"] = 0.0
                    results[f"{k}_std"] = 0.0
            else:
                results[k] = float(arr.mean())
        return results
