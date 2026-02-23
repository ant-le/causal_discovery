from __future__ import annotations

import argparse
import logging
from pathlib import Path

from causal_meta.analysis.utils import generate_all_artifacts

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
    output_dir = Path(args.output_dir) if args.output_dir else (input_dir / "graphics")

    generate_all_artifacts(overview_path, output_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
