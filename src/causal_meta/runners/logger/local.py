import logging
from typing import Any, Dict, List, Optional

from .base import BaseLogger

log = logging.getLogger(__name__)


class LocalLogger(BaseLogger):
    """
    Logs metrics to the standard Python logger and stores them in memory.
    """

    def __init__(self):
        self.history: List[Dict[str, Any]] = []

    def log_metrics(
        self, metrics: Dict[str, float], step: Optional[int] = None
    ) -> None:
        """
        Log metrics to info and append to history.
        """
        self.history.append({"step": step, **metrics})

        # Format for console readability
        metric_str = " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        prefix = f"Step {step}: " if step is not None else ""
        log.info(f"{prefix}{metric_str}")

    def log_hyperparams(self, params: Dict[str, Any]) -> None:
        """
        Log params to info.
        """
        log.info(f"Hyperparameters: {params}")

    def finish(self) -> None:
        """
        No-op for local logger (history is in memory).
        """
        pass
