from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional, Sequence, Set

import torch

from causal_meta.datasets.generators.graphs import (
    ErdosRenyiGenerator,
    SBMGenerator,
    ScaleFreeGenerator,
)
from causal_meta.datasets.generators.mechanisms import (
    LinearMechanismFactory,
    MLPMechanismFactory,
)
from causal_meta.datasets.scm import SCMFamily
from causal_meta.datasets.torch_datasets import MetaFixedDataset, MetaIterableDataset
from causal_meta.datasets.utils import collate_fn_scm, compute_graph_hash
from torch.utils.data import DataLoader

try:  # Optional Hydra/OmegaConf support
    from omegaconf import DictConfig, OmegaConf
except Exception:  # pragma: no cover - optional dependency
    DictConfig = None
    OmegaConf = None


@dataclass
class FamilyConfig:
    """Configuration holder for SCM families."""

    name: str
    graph_type: str
    mech_type: str
    n_nodes: int
    sparsity: Optional[float] = None
    graph_params: Dict[str, Any] = field(default_factory=dict)
    mech_params: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.n_nodes < 1:
            raise ValueError("n_nodes must be positive.")
        if not self.graph_type:
            raise ValueError("graph_type must be provided.")
        if not self.mech_type:
            raise ValueError("mech_type must be provided.")


class CausalMetaModule:
    """Dataset orchestrator that enforces disjoint train/test families."""

    def __init__(
        self,
        train_family_cfg: FamilyConfig | Dict[str, Any],
        test_family_cfg: FamilyConfig | Dict[str, Any],
        seeds_train: Sequence[int],
        seeds_test: Sequence[int],
        *,
        base_seed: int = 0,
        samples_per_task: int = 128,
        cache_test: bool = True,
        safety_checks: bool = True,
        num_workers: int = 0,
        pin_memory: bool = True,
    ) -> None:
        self.train_family_cfg = self._coerce_family_config(train_family_cfg)
        self.test_family_cfg = self._coerce_family_config(test_family_cfg)
        self.seeds_train = list(seeds_train)
        self.seeds_test = list(seeds_test)
        self.base_seed = base_seed
        self.samples_per_task = samples_per_task
        self.cache_test = cache_test
        self.safety_checks = safety_checks
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.train_family: Optional[SCMFamily] = None
        self.test_family: Optional[SCMFamily] = None
        self.train_dataset: Optional[MetaIterableDataset] = None
        self.test_dataset: Optional[MetaFixedDataset] = None
        self.spectral_distance: Optional[float] = None

    def setup(self) -> None:
        """Instantiate datasets, enforce disjointness, and pre-compute stats."""
        self.train_family_cfg.validate()
        self.test_family_cfg.validate()
        self.train_family = self._build_family(self.train_family_cfg)
        self.test_family = self._build_family(self.test_family_cfg)

        test_hashes = self._sample_hashes(self.test_family, self.seeds_test)
        if self.safety_checks:
            self._check_disjoint(self.train_family, test_hashes)
            self.spectral_distance = self._compute_spectral_distance(
                self.train_family, self.test_family
            )
        else:
            self.spectral_distance = None

        self.train_dataset = MetaIterableDataset(
            self.train_family,
            base_seed=self.base_seed,
            samples_per_task=self.samples_per_task,
            forbidden_hashes=test_hashes,
        )
        self.test_dataset = MetaFixedDataset(
            self.test_family,
            seeds=self.seeds_test,
            cache=self.cache_test,
            samples_per_task=self.samples_per_task,
        )

    def train_dataloader(self) -> DataLoader:
        """Return the training DataLoader with GPU optimizations."""
        if self.train_dataset is None:
            self.setup()
        return DataLoader(
            self.train_dataset,  # type: ignore
            batch_size=None,  # Handled by dataset (iterable)
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn_scm,
        )

    def test_dataloader(self) -> DataLoader:
        """Return the test DataLoader."""
        if self.test_dataset is None:
            self.setup()
        return DataLoader(
            self.test_dataset,  # type: ignore
            batch_size=1,  # One task per batch
            num_workers=0,  # No need for workers on fixed test set
            pin_memory=self.pin_memory,
            collate_fn=collate_fn_scm,
        )

    def _coerce_family_config(self, cfg: FamilyConfig | Dict[str, Any]) -> FamilyConfig:
        if OmegaConf is not None and DictConfig is not None and isinstance(cfg, DictConfig):
            cfg = OmegaConf.to_object(cfg)  # type: ignore[arg-type]
        if isinstance(cfg, dict):
            return FamilyConfig(**cfg)
        if isinstance(cfg, FamilyConfig):
            return cfg
        raise TypeError("FamilyConfig must be a dataclass instance or a mapping.")

    def _build_family(self, cfg: FamilyConfig) -> SCMFamily:
        graph_type = cfg.graph_type.lower()
        mech_type = cfg.mech_type.lower()

        if graph_type in {"er", "erdos-renyi", "erdos_renyi"}:
            edge_prob = cfg.sparsity if cfg.sparsity is not None else cfg.graph_params.get("edge_prob")
            if edge_prob is None:
                raise ValueError("Erdos-Renyi generator requires a sparsity/edge_prob parameter.")
            graph_generator = ErdosRenyiGenerator(edge_prob=edge_prob)
        elif graph_type in {"scale_free", "scale-free", "sf"}:
            m_param = int(cfg.graph_params.get("m", 2))
            graph_generator = ScaleFreeGenerator(m=m_param)
        elif graph_type == "sbm":
            try:
                n_blocks = int(cfg.graph_params["n_blocks"])
                p_intra = float(cfg.graph_params["p_intra"])
                p_inter = float(cfg.graph_params["p_inter"])
            except KeyError as exc:
                raise ValueError("SBM generator requires n_blocks, p_intra, and p_inter.") from exc
            graph_generator = SBMGenerator(n_blocks=n_blocks, p_intra=p_intra, p_inter=p_inter)
        else:
            raise ValueError(f"Unsupported graph_type: {cfg.graph_type}")

        if mech_type == "linear":
            weight_scale = float(cfg.mech_params.get("weight_scale", 1.0))
            mechanism_factory = LinearMechanismFactory(weight_scale=weight_scale)
        elif mech_type == "mlp":
            hidden_dim = int(cfg.mech_params.get("hidden_dim", 32))
            mechanism_factory = MLPMechanismFactory(hidden_dim=hidden_dim)
        else:
            raise ValueError(f"Unsupported mech_type: {cfg.mech_type}")

        return SCMFamily(n_nodes=cfg.n_nodes, graph_generator=graph_generator, mechanism_factory=mechanism_factory)

    def _check_disjoint(self, train_family: SCMFamily, test_hashes: Set[str]) -> None:
        probe_seeds = self._probe_seeds(self.seeds_train)
        collisions: Set[str] = set()
        for seed in probe_seeds:
            instance = train_family.sample_task(seed)
            graph_hash = compute_graph_hash(instance.adjacency_matrix)
            if graph_hash in test_hashes:
                collisions.add(graph_hash)
        if collisions:
            warnings.warn(
                f"Detected graph hash collisions between train and test splits: {collisions}. "
                "Consider adjusting seeds or family parameters.",
                RuntimeWarning,
            )

    def _sample_hashes(self, family: SCMFamily, seeds: Sequence[int]) -> Set[str]:
        hashes: Set[str] = set()
        probe = self._probe_seeds(seeds)
        for seed in probe:
            instance = family.sample_task(seed)
            hashes.add(compute_graph_hash(instance.adjacency_matrix))
        return hashes

    def _probe_seeds(self, seeds: Sequence[int], count: int = 3) -> Sequence[int]:
        if seeds:
            return list(seeds)[:count]
        return [self.base_seed + i for i in range(count)]

    def _compute_spectral_distance(self, train_family: SCMFamily, test_family: SCMFamily) -> float:
        train_profile = self._spectral_profile(train_family, self.seeds_train)
        test_profile = self._spectral_profile(test_family, self.seeds_test)
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
