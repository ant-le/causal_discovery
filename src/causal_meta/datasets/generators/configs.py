from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Union

from causal_meta.datasets.generators.graphs import (
    ErdosRenyiGenerator,
    MixtureGraphGenerator,
    SBMGenerator,
    ScaleFreeGenerator,
)
from causal_meta.datasets.generators.mechanisms import (
    LinearMechanismFactory,
    MechanismFactory,
    MixtureMechanismFactory,
    MLPMechanismFactory,
)


class Instantiable(Protocol):
    def instantiate(self) -> Any: ...


@dataclass
class ErdosRenyiConfig:
    edge_prob: Optional[float] = None
    sparsity: Optional[float] = None

    def instantiate(self) -> ErdosRenyiGenerator:
        return ErdosRenyiGenerator(edge_prob=self.edge_prob, sparsity=self.sparsity)


@dataclass
class ScaleFreeConfig:
    m: int = 2

    def instantiate(self) -> ScaleFreeGenerator:
        return ScaleFreeGenerator(m=self.m)


@dataclass
class SBMConfig:
    n_blocks: int
    p_intra: float
    p_inter: float

    def instantiate(self) -> SBMGenerator:
        return SBMGenerator(
            n_blocks=self.n_blocks, p_intra=self.p_intra, p_inter=self.p_inter
        )


@dataclass
class MixtureGraphConfig:
    generators: List[Instantiable]
    weights: List[float]

    def instantiate(self) -> MixtureGraphGenerator:
        return MixtureGraphGenerator(
            generators=[g.instantiate() for g in self.generators], weights=self.weights
        )


@dataclass
class LinearMechanismConfig:
    weight_scale: float = 1.0
    noise_concentration: float = 2.0
    noise_rate: float = 2.0

    def instantiate(self) -> LinearMechanismFactory:
        return LinearMechanismFactory(
            weight_scale=self.weight_scale,
            noise_concentration=self.noise_concentration,
            noise_rate=self.noise_rate,
        )


@dataclass
class MLPMechanismConfig:
    hidden_dim: int = 32

    def instantiate(self) -> MLPMechanismFactory:
        return MLPMechanismFactory(hidden_dim=self.hidden_dim)


@dataclass
class MixtureMechanismConfig:
    factories: List[Instantiable]
    weights: List[float]

    def instantiate(self) -> MixtureMechanismFactory:
        return MixtureMechanismFactory(
            factories=[f.instantiate() for f in self.factories], weights=self.weights
        )


# Unions for Type Hinting
GraphConfig = Union[
    ErdosRenyiConfig, ScaleFreeConfig, SBMConfig, MixtureGraphConfig, Instantiable
]
MechanismConfig = Union[
    LinearMechanismConfig,
    MLPMechanismConfig,
    MixtureMechanismConfig,
    Instantiable,
]


@dataclass
class FamilyConfig:
    """Configuration for a single SCM Family (graph distribution + mechanisms)."""
    name: str
    n_nodes: int
    graph_cfg: GraphConfig
    mech_cfg: MechanismConfig
    
    def validate(self) -> None:
        if self.n_nodes < 1:
            raise ValueError("n_nodes must be positive.")


@dataclass


class DataModuleConfig:


    """Top-level configuration for the CausalMetaModule."""


    train_family: FamilyConfig

    # Support multiple named test families (e.g., {"ood_graph": ..., "ood_mech": ...})


    test_families: Dict[str, FamilyConfig]


    seeds_train: List[int]

    seeds_test: List[int]

    # Optional validation families (default is {"id": train_family} when empty)
    val_families: Dict[str, FamilyConfig] = field(default_factory=dict)

    # Fixed validation seeds (distinct from train/test)
    seeds_val: List[int] = field(default_factory=list)


    base_seed: int = 0


    samples_per_task: int = 128


    cache_val: bool = True


    cache_test: bool = True


    safety_checks: bool = True


    num_workers: int = 0


    pin_memory: bool = True
