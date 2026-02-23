from __future__ import annotations

import logging
from pathlib import Path

from omegaconf import DictConfig

from causal_meta.analysis.utils import EmptyOverviewError, generate_all_artifacts

log = logging.getLogger(__name__)


def run(cfg: DictConfig, output_dir: Path) -> None:
    target_str = cfg.get("analysis", {}).get("target_dir", None)
    target_dir = Path(target_str) if target_str else output_dir

    overview_name = str(cfg.get("analysis", {}).get("overview_name", "overview.json"))
    overview_path = target_dir / overview_name
    graphics_dir = target_dir / "graphics"

    log.info(f"Running analysis on {target_dir}")
    try:
        generate_all_artifacts(overview_path, graphics_dir)
    except FileNotFoundError:
        log.warning(
            f"Missing {overview_name} at {overview_path}. "
            "Generate overview.json for this multirun first, then rerun analysis."
        )
        return
    except EmptyOverviewError:
        log.warning("DataFrame is empty after loading/aggregation.")
        return

    log.info(f"Analysis artifacts saved to {graphics_dir}")
