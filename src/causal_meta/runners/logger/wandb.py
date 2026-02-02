from __future__ import annotations

import logging
from types import ModuleType
from typing import Any, Dict, Mapping, Optional

from .base import BaseLogger

log = logging.getLogger(__name__)


def _get(obj: Any, key: str, default: Any) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    if hasattr(obj, "get"):
        try:
            return obj.get(key, default)  # type: ignore[no-any-return]
        except Exception:
            pass
    return getattr(obj, key, default)


def wandb_enabled(cfg: Any) -> bool:
    logger_cfg = _get(cfg, "logger", {})
    wandb_cfg = _get(logger_cfg, "wandb", {})
    return bool(_get(wandb_cfg, "enabled", False))


def import_wandb_if_enabled(enabled: bool) -> ModuleType | None:
    if not enabled:
        return None
    try:
        import wandb  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "W&B logging is enabled but 'wandb' is not installed. "
            "Install it (e.g., `pip install wandb`) or disable it via `logger.wandb.enabled=false`."
        ) from exc
    return wandb


class WandbLogger(BaseLogger):
    """
    WandB implementation of the BaseLogger.
    """

    def __init__(self, cfg: Any, output_dir: Optional[str] = None):
        self.wandb_module = import_wandb_if_enabled(True)
        if self.wandb_module is None:
            raise RuntimeError("WandbLogger initialized but wandb import failed.")

        from omegaconf import OmegaConf

        wandb_cfg = _get(_get(cfg, "logger", {}), "wandb", {})

        # Initialize run
        self.run = self.wandb_module.init(
            project=_get(wandb_cfg, "project", "causal_meta"),
            entity=_get(wandb_cfg, "entity", None),
            name=_get(wandb_cfg, "name", _get(cfg, "name", "experiment")),
            config=(
                OmegaConf.to_container(cfg, resolve=True)
                if hasattr(cfg, "logger")
                else cfg
            ),  # handle raw dict or DictConfig
            tags=_get(wandb_cfg, "tags", []),
            mode=_get(wandb_cfg, "mode", "offline"),
            dir=output_dir,
        )

    def log_metrics(
        self, metrics: Dict[str, float], step: Optional[int] = None
    ) -> None:
        if self.run:
            if step is not None:
                self.wandb_module.log(metrics, step=step)
            else:
                self.wandb_module.log(metrics)

    def log_hyperparams(self, params: Dict[str, Any]) -> None:
        if self.run:
            self.wandb_module.config.update(params, allow_val_change=True)

    def finish(self) -> None:
        if self.run:
            self.wandb_module.finish()
