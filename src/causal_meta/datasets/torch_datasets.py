from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Set, Tuple

import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from causal_meta.datasets.scm import SCMFamily, SCMInstance
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
                x = instance.sample(self.samples_per_task)
            yield x, instance.adjacency_matrix
            seed += stride

    # TODO: add validation logic (after a specific number of samples?)
    # can have same logic as testing with id and ood buts needs 
    # different seeds (distinct from test set)

class MetaFixedDataset(Dataset):
    """Fixed-seed SCM dataset with optional instance and data caching."""

    def __init__(
        self,
        family: SCMFamily,
        seeds: Sequence[int],
        cache: bool = True,
        samples_per_task: int = 128,
    ) -> None:
        self.family = family
        self.seeds = list(seeds)
        self.cache_instances = cache
        self.samples_per_task = samples_per_task
        # Cache stores Tuple[SCMInstance, torch.Tensor]
        self._cache: Dict[int, Tuple[SCMInstance, torch.Tensor]] = {}

    def __len__(self) -> int:
        return len(self.seeds)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seed = int(self.seeds[idx])

        if self.cache_instances and seed in self._cache:
            instance, x = self._cache[seed]
        else:
            instance = self.family.sample_task(seed)
            # Deterministic sampling based on the seed
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                x = instance.sample(self.samples_per_task)
            
            if self.cache_instances:
                self._cache[seed] = (instance, x)

        return x, instance.adjacency_matrix

    # TODO: add logic to safe cache experiments folder in the end
