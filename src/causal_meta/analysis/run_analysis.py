from __future__ import annotations

import argparse
import logging
from pathlib import Path

from causal_meta.analysis.plots.results import (generate_performance_figure,
                                                generate_structural_figure)
from causal_meta.analysis.tables.results import generate_robustness_table
from causal_meta.analysis.utils import load_overview_json

log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate plots/tables from a multirun directory containing overview.json"
        )
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Path to multirun root directory (must contain overview.json)",
    )
    parser.add_argument(
        "--overview-name",
        type=str,
        default="overview.json",
        help="Overview filename inside input_dir",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=("Output directory for artifacts. Defaults to <input_dir>/graphics"),
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    overview_path = input_dir / str(args.overview_name)
    if not overview_path.exists():
        raise FileNotFoundError(
            f"Missing overview file: {overview_path}. "
            "Generate overview.json first, then rerun this command."
        )

    output_dir = Path(args.output_dir) if args.output_dir else (input_dir / "graphics")
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Loading overview from {overview_path}")
    df = load_overview_json(str(overview_path))
    if df.empty:
        raise RuntimeError(f"No data found in {overview_path}")

    log.info(f"Writing artifacts to {output_dir}")
    generate_structural_figure(df, output_dir / "structural_metrics.png")
    generate_performance_figure(df, output_dir / "performance_metrics.png")
    generate_robustness_table(df, output_dir / "robustness_table.tex")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
