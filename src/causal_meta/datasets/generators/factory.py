from __future__ import annotations

from typing import Any, Dict, List, Mapping, Protocol

from causal_meta.datasets.generators import configs

# Optional Hydra support
try:
    from hydra.utils import instantiate as hydra_instantiate
    from omegaconf import DictConfig, OmegaConf
except ImportError:
    hydra_instantiate = None
    DictConfig = None
    OmegaConf = None


class Instantiable(Protocol):
    def instantiate(self) -> Any: ...


class _HydraConfigWrapper:
    def __init__(self, cfg: Mapping[str, Any]) -> None:
        self.cfg = cfg

    def instantiate(self) -> Any:
        if hydra_instantiate is None:
            raise RuntimeError("Hydra is not installed but '_target_' was found.")
        return hydra_instantiate(self.cfg, _recursive_=True)


class _DirectObjectWrapper:
    def __init__(self, obj: Any) -> None:
        self.obj = obj

    def instantiate(self) -> Any:
        return self.obj


def _coerce_dict(cfg: Any) -> Any:
    if DictConfig is not None and isinstance(cfg, DictConfig):
        from omegaconf import OmegaConf as _OmegaConf

        return _OmegaConf.to_container(cfg, resolve=True)
    return cfg


def _exclude_type(d: Mapping[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in d.items() if k != "type"}


GRAPH_CONFIG_MAP = {
    "er": configs.ErdosRenyiConfig,
    "sf": configs.ScaleFreeConfig,
    "scale_free": configs.ScaleFreeConfig,
    "sbm": configs.SBMConfig,
    "mixture": configs.MixtureGraphConfig,
}

MECHANISM_CONFIG_MAP = {
    "linear": configs.LinearMechanismConfig,
    "mlp": configs.MLPMechanismConfig,
    "mixture": configs.MixtureMechanismConfig,
    "square": configs.SquareMechanismConfig,
    "periodic": configs.PeriodicMechanismConfig,
    "logistic": configs.LogisticMapMechanismConfig,
    "logistic_map": configs.LogisticMapMechanismConfig,
    "gp": configs.GPMechanismConfig,
    "pnl": configs.PNLMechanismConfig,
}


def load_graph_config(cfg: Any) -> configs.GraphConfig:
    cfg = _coerce_dict(cfg)

    if hasattr(cfg, "instantiate"):
        return cfg  # type: ignore

    if callable(cfg) and not isinstance(cfg, Mapping):
        return _DirectObjectWrapper(cfg)

    if not isinstance(cfg, Mapping):
        raise TypeError(
            f"Graph config must be a mapping, callable, or config object. Got {type(cfg)}"
        )

    if "_target_" in cfg:
        return _HydraConfigWrapper(cfg)

    type_name = cfg.get("type")
    if not type_name:
        raise ValueError("Graph config must contain a 'type' or '_target_' key.")

    if type_name == "mixture":
        kwargs = _exclude_type(cfg)
        if "generators" not in kwargs:
            raise ValueError("Mixture graph config must contain 'generators' list.")
        kwargs["generators"] = [load_graph_config(g) for g in kwargs["generators"]]
        return configs.MixtureGraphConfig(**kwargs)

    config_cls = GRAPH_CONFIG_MAP.get(str(type_name))
    if config_cls is None:
        raise ValueError(f"Unknown graph generator type: '{type_name}'")

    return config_cls(**_exclude_type(cfg))


def load_mechanism_config(cfg: Any) -> configs.MechanismConfig:
    cfg = _coerce_dict(cfg)

    if hasattr(cfg, "instantiate"):
        return cfg  # type: ignore

    if callable(cfg) and not isinstance(cfg, Mapping):
        return _DirectObjectWrapper(cfg)

    if not isinstance(cfg, Mapping):
        raise TypeError(
            f"Mechanism config must be a mapping, callable, or config object. Got {type(cfg)}"
        )

    if "_target_" in cfg:
        return _HydraConfigWrapper(cfg)

    type_name = cfg.get("type")
    if not type_name:
        raise ValueError("Mechanism config must contain a 'type' or '_target_' key.")

    if type_name == "mixture":
        kwargs = _exclude_type(cfg)
        if "factories" not in kwargs:
            raise ValueError("Mixture mechanism config must contain 'factories' list.")
        kwargs["factories"] = [load_mechanism_config(f) for f in kwargs["factories"]]
        return configs.MixtureMechanismConfig(**kwargs)

    # PNL supports a nested inner mechanism config.
    # Coerce it recursively so downstream code can call `.instantiate()`.
    if type_name in {"pnl"}:
        kwargs = _exclude_type(cfg)
        inner = kwargs.get("inner_config")
        if inner is not None:
            kwargs["inner_config"] = load_mechanism_config(inner)
        return configs.PNLMechanismConfig(**kwargs)

    config_cls = MECHANISM_CONFIG_MAP.get(str(type_name))
    if config_cls is None:
        raise ValueError(f"Unknown mechanism factory type: '{type_name}'")

    return config_cls(**_exclude_type(cfg))


def load_family_config(cfg: Any) -> configs.FamilyConfig:
    cfg = _coerce_dict(cfg)

    if isinstance(cfg, configs.FamilyConfig):
        return cfg

    if not isinstance(cfg, Mapping):
        raise TypeError("Family config must be a dict or FamilyConfig object.")

    graph_cfg = cfg.get("graph_cfg", cfg.get("graph"))
    mech_cfg = cfg.get("mech_cfg", cfg.get("mech"))

    if graph_cfg is None or mech_cfg is None:
        raise ValueError("Family config must provide 'graph_cfg' and 'mech_cfg'.")

    return configs.FamilyConfig(
        name=str(cfg.get("name", "")),
        n_nodes=int(cfg["n_nodes"]),
        graph_cfg=load_graph_config(graph_cfg),
        mech_cfg=load_mechanism_config(mech_cfg),
    )


def load_data_module_config(cfg: Any) -> configs.DataModuleConfig:
    cfg = _coerce_dict(cfg)

    if isinstance(cfg, configs.DataModuleConfig):
        return cfg

    if not isinstance(cfg, Mapping):
        raise TypeError("DataModule config must be a dict or DataModuleConfig object.")

    train_family = load_family_config(cfg["train_family"])

    test_families_raw = cfg.get("test_families")
    if test_families_raw is None:
        raise ValueError("Config must contain 'test_families'.")

    if not isinstance(test_families_raw, Mapping):
        raise TypeError("'test_families' must be a dictionary of configs.")

    test_families = {
        name: load_family_config(sub_cfg) for name, sub_cfg in test_families_raw.items()
    }

    val_families_raw = cfg.get("val_families")
    if val_families_raw is None:
        val_families: Dict[str, configs.FamilyConfig] = {}
    else:
        if not isinstance(val_families_raw, Mapping):
            raise TypeError("'val_families' must be a dictionary of configs.")
        val_families = {
            name: load_family_config(sub_cfg)
            for name, sub_cfg in val_families_raw.items()
        }

    allowed_keys = {
        "seeds_test",
        "seeds_val",
        "base_seed",
        "samples_per_task",
        "safety_checks",
        "num_workers",
        "pin_memory",
        "normalize_data",
        "batch_size_train",
        "batch_size_val",
        "batch_size_test",
        "batch_size_test_interventional",
    }

    kwargs = {k: v for k, v in cfg.items() if k in allowed_keys}

    if "seeds_test" not in kwargs or "seeds_val" not in kwargs:
        raise ValueError("Config must contain 'seeds_test' and 'seeds_val'.")

    return configs.DataModuleConfig(
        train_family=train_family,
        val_families=val_families,
        test_families=test_families,
        **kwargs,
    )
