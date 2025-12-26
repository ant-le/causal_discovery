from __future__ import annotations

import warnings
from typing import Any, Dict, Optional, Sequence, Set

import torch
from torch.utils.data import DataLoader

from causal_meta.datasets.generators.configs import DataModuleConfig, FamilyConfig
from causal_meta.datasets.generators.factory import load_data_module_config
from causal_meta.datasets.scm import SCMFamily
from causal_meta.datasets.torch_datasets import MetaFixedDataset, MetaIterableDataset
from causal_meta.datasets.utils import collate_fn_scm, compute_graph_hash


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
        
        self.spectral_distances: Dict[str, float] = {}

    def setup(self) -> None:
        """Instantiate datasets, enforce disjointness, and pre-compute stats."""
        self.train_family = self._build_family(self.config.train_family)

        val_family_cfgs = self.config.val_families or {"id": self.config.train_family}
        self.val_families = {
            name: self._build_family(cfg)
            for name, cfg in val_family_cfgs.items()
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
                all_val_hashes.update(self._sample_hashes(family, self.config.seeds_val))

        all_test_hashes: Set[str] = set()
        for family in self.test_families.values():
             all_test_hashes.update(self._sample_hashes(family, self.config.seeds_test))

        all_reserved_hashes = all_val_hashes | all_test_hashes
        
        if self.config.safety_checks:
            self._check_disjoint(self.train_family, all_reserved_hashes)

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
                cache=self.config.cache_val,
                samples_per_task=self.config.samples_per_task,
            )
            for name, family in self.val_families.items()
        }
        
        self.test_datasets = {
            name: MetaFixedDataset(
                family,
                seeds=self.config.seeds_test,
                cache=self.config.cache_test,
                samples_per_task=self.config.samples_per_task,
            )
            for name, family in self.test_families.items()
        }

    def train_dataloader(self) -> DataLoader:
        """Return the training DataLoader with GPU optimizations."""
        if self.train_dataset is None:
            self.setup()
        return DataLoader(
            self.train_dataset,  # type: ignore
            batch_size=1,  # One task per batch (required for collate_fn_scm)
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            collate_fn=collate_fn_scm,
        )

    def test_dataloader(self) -> Dict[str, DataLoader]:
        """Return a dictionary of test DataLoaders."""
        if not self.test_datasets:
            self.setup()
        
        loaders = {}
        is_distributed = torch.distributed.is_available() and torch.distributed.is_initialized()

        for name, dataset in self.test_datasets.items():
            sampler = None
            if is_distributed:
                sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)
            
            loaders[name] = DataLoader(
                dataset,
                batch_size=1,  # One task per batch
                num_workers=0,  # No need for workers on fixed test set
                pin_memory=self.config.pin_memory,
                collate_fn=collate_fn_scm,
                sampler=sampler
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
        is_distributed = torch.distributed.is_available() and torch.distributed.is_initialized()

        for name, dataset in self.val_datasets.items():
            sampler = None
            if is_distributed:
                sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)

            loaders[name] = DataLoader(
                dataset,
                batch_size=1,  # One task per batch
                num_workers=0,  # No need for workers on fixed validation set
                pin_memory=self.config.pin_memory,
                collate_fn=collate_fn_scm,
                sampler=sampler
            )
        return loaders

    def _check_disjoint(self, train_family: SCMFamily, test_hashes: Set[str]) -> None:
        probe_seeds = self._probe_seeds(self.config.seeds_train)
        collisions: Set[str] = set()
        for seed in probe_seeds:
            instance = train_family.sample_task(seed)
            graph_hash = compute_graph_hash(instance.adjacency_matrix)
            if graph_hash in test_hashes:
                collisions.add(graph_hash)
        if collisions:
            warnings.warn(
                f"Detected graph hash collisions between train and held-out (val/test) splits: {collisions}. "
                "Consider adjusting seeds or family parameters.",
                RuntimeWarning,
            )

    def _sample_hashes(self, family: SCMFamily, seeds: Sequence[int]) -> Set[str]:
        hashes: Set[str] = set()
        probe = list(seeds) if seeds else self._probe_seeds(seeds)
        for seed in probe:
            instance = family.sample_task(seed)
            hashes.add(compute_graph_hash(instance.adjacency_matrix))
        return hashes

    def _probe_seeds(self, seeds: Sequence[int], count: int = 3) -> Sequence[int]:
        if seeds:
            return list(seeds)[:count]
        return [self.config.base_seed + i for i in range(count)]

    def _compute_spectral_distance(self, train_family: SCMFamily, test_family: SCMFamily) -> float:
        train_profile = self._spectral_profile(train_family, self.config.seeds_train)
        test_profile = self._spectral_profile(test_family, self.config.seeds_test)
        min_len = min(train_profile.numel(), test_profile.numel())
        if min_len == 0:
            return 0.0
        distance = torch.mean(torch.abs(train_profile[:min_len] - test_profile[:min_len]))
        return float(distance.item())

    def _spectral_profile(self, family: SCMFamily, seeds: Sequence[int], count: int = 3) -> torch.Tensor:
        probe = self._probe_seeds(seeds, count=count)
        eigenvalues = []
        for seed in probe:
            adj = family.sample_task(seed).adjacency_matrix
            sym_adj = (adj + adj.T) / 2.0
            vals = torch.linalg.eigvalsh(sym_adj)
            eigenvalues.append(vals)
        if not eigenvalues:
            return torch.tensor([])
        return torch.stack(eigenvalues, dim=0).mean(dim=0)

    def _build_family(self, cfg: FamilyConfig) -> SCMFamily:
        graph_generator = cfg.graph_cfg.instantiate()
        mechanism_factory = cfg.mech_cfg.instantiate()
        return SCMFamily(
            n_nodes=cfg.n_nodes,
            graph_generator=graph_generator,
            mechanism_factory=mechanism_factory,
        )
