from __future__ import annotations

import warnings
from functools import partial
from typing import Any, Dict, Optional, Sequence, Set

import torch
from torch.utils.data import DataLoader

from causal_meta.datasets.generators.configs import (DataModuleConfig,
                                                     FamilyConfig)
from causal_meta.datasets.generators.factory import load_data_module_config
from causal_meta.datasets.scm import SCMFamily
from causal_meta.datasets.torch_datasets import (MetaFixedDataset,
                                                 MetaInterventionalDataset,
                                                 MetaIterableDataset)
from causal_meta.datasets.utils import (collate_fn_interventional,
                                        collate_fn_scm, compute_graph_hash)
from causal_meta.datasets.utils.sampling import NoPaddingDistributedSampler


class CausalMetaModule:
    """Dataset orchestrator that enforces disjoint train/val/test families."""

    @classmethod
    def from_config(cls, cfg: Any) -> CausalMetaModule:
        """Build a module from a configuration object (Dict, Hydra, or DataModuleConfig)."""
        data_module_config = load_data_module_config(cfg)
        return cls(data_module_config)

    def __init__(self, config: DataModuleConfig) -> None:
        """Initialize the module with a DataModuleConfig object."""
        self.config = config

        self.train_family: Optional[SCMFamily] = None
        self.val_families: Dict[str, SCMFamily] = {}
        self.test_families: Dict[str, SCMFamily] = {}

        self.train_dataset: Optional[MetaIterableDataset] = None
        self.val_datasets: Dict[str, MetaFixedDataset] = {}
        self.test_datasets: Dict[str, MetaFixedDataset] = {}
        self.test_interventional_datasets: Dict[str, MetaInterventionalDataset] = {}

        self.spectral_distances: Dict[str, float] = {}

    def setup(self) -> None:
        """Instantiate datasets, enforce disjointness, and pre-compute stats."""
        self.train_family = self._build_family(self.config.train_family)

        val_family_cfgs = self.config.val_families or {"id": self.config.train_family}
        self.val_families = {
            name: self._build_family(cfg) for name, cfg in val_family_cfgs.items()
        }

        # Build all test families
        self.test_families = {
            name: self._build_family(cfg)
            for name, cfg in self.config.test_families.items()
        }

        # Collect hashes from validation and test sets for disjointness check
        all_val_hashes: Set[str] = set()
        if self.config.seeds_val:
            for family in self.val_families.values():
                all_val_hashes.update(
                    self._sample_hashes(family, self.config.seeds_val)
                )

        all_test_hashes: Set[str] = set()
        for family in self.test_families.values():
            all_test_hashes.update(self._sample_hashes(family, self.config.seeds_test))

        all_reserved_hashes = all_val_hashes | all_test_hashes

        if self.config.safety_checks:
            val_test_collisions = all_val_hashes & all_test_hashes
            if val_test_collisions:
                warnings.warn(
                    f"Detected graph hash collisions between validation and test splits: {val_test_collisions}. "
                    "Consider adjusting seeds or family parameters.",
                    RuntimeWarning,
                )

            # Compute distances for each test family
            self.spectral_distances = {
                name: self._compute_spectral_distance(self.train_family, family)
                for name, family in self.test_families.items()
            }
        else:
            self.spectral_distances = {}

        self.train_dataset = MetaIterableDataset(
            self.train_family,
            base_seed=self.config.base_seed,
            samples_per_task=self.config.samples_per_task,
            forbidden_hashes=all_reserved_hashes,
        )

        self.val_datasets = {
            name: MetaFixedDataset(
                family,
                seeds=self.config.seeds_val,
                samples_per_task=self.config.samples_per_task,
            )
            for name, family in self.val_families.items()
        }

        self.test_datasets = {
            name: MetaFixedDataset(
                family,
                seeds=self.config.seeds_test,
                samples_per_task=self.config.samples_per_task,
            )
            for name, family in self.test_families.items()
        }

        self.test_interventional_datasets = {
            name: MetaInterventionalDataset(
                family,
                seeds=self.config.seeds_test,
                intervention_value=0.0,
                samples_per_task=self.config.samples_per_task,
            )
            for name, family in self.test_families.items()
        }

    def train_dataloader(self) -> DataLoader:
        """Return the training DataLoader with GPU optimizations."""
        if self.train_dataset is None:
            self.setup()

        collate_fn = partial(collate_fn_scm, normalize=self.config.normalize_data)

        batch_size = int(getattr(self.config, "batch_size_train", 1))
        if batch_size < 1:
            raise ValueError("data.batch_size_train must be >= 1")

        return DataLoader(
            self.train_dataset,  # type: ignore
            batch_size=batch_size,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            collate_fn=collate_fn,
        )

    def test_dataloader(self) -> Dict[str, DataLoader]:
        """Return a dictionary of test DataLoaders."""
        if not self.test_datasets:
            self.setup()

        loaders = {}
        is_distributed = (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        )

        collate_fn = partial(collate_fn_scm, normalize=self.config.normalize_data)

        batch_size = int(getattr(self.config, "batch_size_test", 1))
        if batch_size < 1:
            raise ValueError("data.batch_size_test must be >= 1")

        for name, dataset in self.test_datasets.items():
            sampler = None
            if is_distributed:
                sampler = NoPaddingDistributedSampler(dataset, shuffle=False)

            loaders[name] = DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=self.config.num_workers,  # Use configured workers
                pin_memory=self.config.pin_memory,
                collate_fn=collate_fn,
                sampler=sampler,
            )
        return loaders

    def test_interventional_dataloader(self) -> Dict[str, DataLoader]:
        """Return a dictionary of interventional test DataLoaders."""
        if not self.test_interventional_datasets:
            self.setup()

        loaders = {}
        is_distributed = (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        )

        collate_fn = partial(
            collate_fn_interventional, normalize=self.config.normalize_data
        )

        batch_size = int(getattr(self.config, "batch_size_test_interventional", 1))
        if batch_size < 1:
            raise ValueError("data.batch_size_test_interventional must be >= 1")

        for name, dataset in self.test_interventional_datasets.items():
            sampler = None
            if is_distributed:
                sampler = NoPaddingDistributedSampler(dataset, shuffle=False)

            loaders[name] = DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory,
                collate_fn=collate_fn,
                sampler=sampler,
            )
        return loaders

    def val_dataloader(self) -> Dict[str, DataLoader]:
        """Return a dictionary of validation DataLoaders."""
        if not self.val_datasets:
            self.setup()

        if not self.config.seeds_val:
            raise ValueError(
                "Validation split is not configured. Provide non-empty 'seeds_val' in the data config."
            )

        loaders = {}
        is_distributed = (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        )

        collate_fn = partial(collate_fn_scm, normalize=self.config.normalize_data)

        batch_size = int(getattr(self.config, "batch_size_val", 1))
        if batch_size < 1:
            raise ValueError("data.batch_size_val must be >= 1")

        for name, dataset in self.val_datasets.items():
            sampler = None
            if is_distributed:
                sampler = NoPaddingDistributedSampler(dataset, shuffle=False)

            loaders[name] = DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=self.config.num_workers,  # Use configured workers
                pin_memory=self.config.pin_memory,
                collate_fn=collate_fn,
                sampler=sampler,
            )
        return loaders

    def _sample_hashes(self, family: SCMFamily, seeds: Sequence[int]) -> Set[str]:
        """
        Compute a set of graph hashes for a given list of seeds to ensure disjointness.
        """
        hashes: Set[str] = set()
        # Use provided seeds or probe a default range if empty
        seeds_to_probe = list(seeds) if seeds else self._probe_seeds(seeds)

        for seed in seeds_to_probe:
            adjacency_matrix = family.sample_graph(seed)
            hashes.add(compute_graph_hash(adjacency_matrix))
        return hashes

    def _probe_seeds(self, seeds: Sequence[int], count: int = 3) -> Sequence[int]:
        if seeds:
            return list(seeds)[:count]
        return [self.config.base_seed + i for i in range(count)]

    def _compute_spectral_distance(
        self, train_family: SCMFamily, test_family: SCMFamily
    ) -> float:
        """
        Compute the L1 distance between the average spectral profiles of two families.
        """
        train_profile = self._spectral_profile(train_family, [])
        test_profile = self._spectral_profile(test_family, self.config.seeds_test)

        # Ensure profiles are compatible in length (min length)
        min_len = min(train_profile.numel(), test_profile.numel())
        if min_len == 0:
            return 0.0

        distance = torch.mean(
            torch.abs(train_profile[:min_len] - test_profile[:min_len])
        )
        return float(distance.item())

    def _spectral_profile(
        self, family: SCMFamily, seeds: Sequence[int], count: int = 3
    ) -> torch.Tensor:
        """
        Compute the average sorted eigenvalues (spectral profile) of graphs sampled from a family.
        """
        seeds_to_probe = self._probe_seeds(seeds, count=count)
        all_eigenvalues = []

        for seed in seeds_to_probe:
            adjacency_matrix = family.sample_graph(seed)
            # Symmetrize to get real eigenvalues: A_sym = (A + A.T) / 2
            symmetric_adj = (adjacency_matrix + adjacency_matrix.T) / 2.0
            eigenvalues = torch.linalg.eigvalsh(symmetric_adj)
            all_eigenvalues.append(eigenvalues)

        if not all_eigenvalues:
            return torch.tensor([])

        # Stack and average across the sampled graphs
        return torch.stack(all_eigenvalues, dim=0).mean(dim=0)

    def _build_family(self, cfg: FamilyConfig) -> SCMFamily:
        graph_generator = cfg.graph_cfg.instantiate()
        mechanism_factory = cfg.mech_cfg.instantiate()
        return SCMFamily(
            n_nodes=cfg.n_nodes,
            graph_generator=graph_generator,
            mechanism_factory=mechanism_factory,
        )
