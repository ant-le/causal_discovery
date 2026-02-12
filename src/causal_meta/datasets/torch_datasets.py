from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Sequence, Set, Tuple

import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from causal_meta.datasets.scm import SCMFamily
from causal_meta.datasets.utils import compute_graph_hash


def _distributed_context() -> Tuple[int, int]:
    """Return rank and world size if torch.distributed is initialized."""
    rank = 0
    world_size = 1
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
    except Exception:
        rank = 0
        world_size = 1
    return rank, world_size


class MetaIterableDataset(IterableDataset):
    """Infinite stream of SCM samples with rank/worker-aware seeding."""

    def __init__(
        self,
        family: SCMFamily,
        base_seed: int,
        samples_per_task: int = 128,
        forbidden_hashes: Optional[Iterable[str]] = None,
    ) -> None:
        super().__init__()
        self.family = family
        self.base_seed = int(base_seed)
        self.samples_per_task = samples_per_task
        self.forbidden_hashes: Set[str] = set(forbidden_hashes or [])

    def __iter__(self):
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1
        rank, world_size = _distributed_context()
        stride = max(1, num_workers * world_size)

        seed = self.base_seed + rank * num_workers + worker_id
        while True:
            instance = self.family.sample_task(seed)
            if self.forbidden_hashes:
                graph_hash = compute_graph_hash(instance.adjacency_matrix)
                if graph_hash in self.forbidden_hashes:
                    seed += stride
                    continue

            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                data = instance.sample(self.samples_per_task)

            yield {
                "seed": int(seed),
                "data": data,
                "adjacency": instance.adjacency_matrix,
            }
            seed += stride


class MetaFixedDataset(Dataset):
    """
    Fixed-seed SCM dataset for deterministic validation/testing.

    This dataset acts as a map-style dataset where each index corresponds to a specific
    deterministic seed from the provided list.
    """

    def __init__(
        self,
        family: SCMFamily,
        seeds: Sequence[int],
        samples_per_task: int = 128,
    ) -> None:
        self.family = family
        self.seeds = list(seeds)
        self.samples_per_task = samples_per_task

    def __len__(self) -> int:
        return len(self.seeds)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Generates and returns an SCM task for the given index.

        Returns:
            dict: {
                "seed": int,
                "data": torch.Tensor (samples),
                "adjacency": torch.Tensor (adjacency matrix)
            }
        """
        seed = int(self.seeds[idx])

        instance = self.family.sample_task(seed)
        # Deterministic sampling based on the seed
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            data = instance.sample(self.samples_per_task)

        adjacency = instance.adjacency_matrix.to(dtype=torch.float32)

        return {"seed": seed, "data": data, "adjacency": adjacency}


class MetaInterventionalDataset(Dataset):
    """
    Fixed-seed SCM dataset that provides both observational and interventional data.
    Performs single-node interventions on all nodes.
    """

    def __init__(
        self,
        family: SCMFamily,
        seeds: Sequence[int],
        intervention_value: float = 0.0,
        samples_per_task: int = 128,
    ) -> None:
        self.family = family
        self.seeds = list(seeds)
        self.intervention_value = intervention_value
        self.samples_per_task = samples_per_task

    def __len__(self) -> int:
        return len(self.seeds)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        seed = int(self.seeds[idx])

        instance = self.family.sample_task(seed)
        n_nodes = instance.n_nodes

        # 1. Observational Data
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            x_obs = instance.sample(self.samples_per_task)

        obs_data = {"data": x_obs, "adjacency": instance.adjacency_matrix}

        # 2. Interventional Data (Single-node on all nodes)
        interventions = []
        for target_node in range(n_nodes):
            # Do(X_target = val)
            mutilated_instance = instance.do({target_node: self.intervention_value})

            # Sample from mutilated graph
            # We use a derived seed for reproducibility of interventions
            int_seed = seed + (target_node + 1) * 1000
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int_seed)
                x_int = mutilated_instance.sample(self.samples_per_task)

            interventions.append(
                {
                    "target": target_node,
                    "value": self.intervention_value,
                    "data": x_int,
                    "adjacency": mutilated_instance.adjacency_matrix,
                }
            )

        result = {
            "observational": obs_data,
            "interventions": interventions,
            "seed": seed,
        }

        return result
