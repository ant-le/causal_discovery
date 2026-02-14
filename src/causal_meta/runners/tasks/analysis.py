from __future__ import annotations

import logging
from pathlib import Path

from omegaconf import DictConfig

from causal_meta.analysis.plots.results import (generate_performance_figure,
                                                generate_structural_figure)
from causal_meta.analysis.tables.results import generate_robustness_table
from causal_meta.analysis.utils import load_overview_json

log = logging.getLogger(__name__)


def run(cfg: DictConfig, output_dir: Path) -> None:
    target_str = cfg.get("analysis", {}).get("target_dir", None)
    if target_str:
        target_dir = Path(target_str)
    else:
        target_dir = output_dir

    log.info(f"Running analysis on {target_dir}")

    overview_name = str(cfg.get("analysis", {}).get("overview_name", "overview.json"))
    overview_path = target_dir / overview_name
    if not overview_path.exists():
        log.warning(
            f"Missing {overview_name} at {overview_path}. "
            "Generate overview.json for this multirun first, then rerun analysis."
        )
        return

    log.info(f"Loading overview from {overview_path}")
    plot_df = load_overview_json(str(overview_path))

    if plot_df.empty:
        log.warning("DataFrame is empty after loading/aggregation.")
        return

    # 3. Generate Artifacts
    graphics_dir = target_dir / "graphics"
    graphics_dir.mkdir(parents=True, exist_ok=True)

    log.info("Generating Structural Metrics Figure...")
    generate_structural_figure(plot_df, graphics_dir / "structural_metrics.png")

    log.info("Generating Performance Metrics Figure...")
    generate_performance_figure(plot_df, graphics_dir / "performance_metrics.png")

    log.info("Generating Robustness Table...")
    generate_robustness_table(plot_df, graphics_dir / "robustness_table.tex")

    log.info(f"Analysis artifacts saved to {graphics_dir}")
