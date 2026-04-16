from __future__ import annotations

import argparse
import logging
from pathlib import Path

from causal_meta.analysis.appendix.generator import generate_appendix_artifacts
from causal_meta.analysis.common import thesis as thesis_common
from causal_meta.analysis.methodology_figures import generate_all_methodology_figures
from causal_meta.analysis.plots.results import (
    generate_event_probability_bar,
    generate_posterior_diagnostic_violins,
)
from causal_meta.analysis.generalisation import plots as gen_plots
from causal_meta.analysis.generalisation import tables as gen_tables
from causal_meta.analysis.transfer.transfer import generate_transfer_figure
from causal_meta.analysis.diagnostics import posterior as diag_posterior
from causal_meta.analysis.uncertainty import plots as unc_plots
from causal_meta.analysis.utils import EmptyAnalysisDataError

log = logging.getLogger(__name__)


def run_thesis_analysis(
    *,
    input_root: Path,
    thesis_root: Path,
    strict: bool = True,
    skip_posterior: bool = False,
) -> Path:
    """Run the curated thesis analysis pipeline, writing directly to graphics/."""

    selected_runs = thesis_common.resolve_thesis_run_directories(input_root)
    run_dirs = [run.run_dir for run in selected_runs]

    # Output directories under graphics/<chapter>/
    results_dir = thesis_root / thesis_common.GRAPHICS_RESULTS
    results_dir.mkdir(parents=True, exist_ok=True)
    appendix_a_dir = thesis_root / thesis_common.GRAPHICS_APPENDIX_A
    appendix_a_dir.mkdir(parents=True, exist_ok=True)
    appendix_e_dir = thesis_root / thesis_common.GRAPHICS_APPENDIX_E
    appendix_e_dir.mkdir(parents=True, exist_ok=True)

    # ── Methodology figures (Chapter 4) ──────────────────────────────
    methodology_dir = thesis_root / thesis_common.GRAPHICS_METHODOLOGY
    generate_all_methodology_figures(methodology_dir)

    summary_df = thesis_common.prepare_summary_dataframe(run_dirs)
    raw_df = thesis_common.prepare_raw_dataframe(run_dirs)

    gen_plots.generate_results_anchor_table(raw_df, results_dir / "results_anchor.tex")

    # ── RQ2: Degradation heatmap ─────────────────────────────────────
    try:
        gen_plots.generate_degradation_heatmap(
            raw_df,
            output_path=results_dir / "degradation_heatmap.pdf",
        )
    except Exception:
        if strict:
            raise
        log.warning("Degradation heatmap failed.", exc_info=True)

    for shift_key in ("graph", "mechanism", "noise"):
        fig_name = f"shift_{shift_key}.pdf"
        try:
            gen_plots.generate_shift_figure(
                raw_df,
                shift_axis=shift_key,
                output_path=results_dir / fig_name,
            )
        except EmptyAnalysisDataError:
            if strict:
                raise
            log.warning("Shift figure for '%s' skipped (no data).", shift_key)

    # ── Compound shift + stress-test (merged) ────────────────────────
    try:
        gen_plots.generate_compound_and_stress_figure(
            raw_df,
            output_path=results_dir / "shift_compound.pdf",
        )
    except Exception:
        if strict:
            raise
        log.warning("Compound + stress-test figure failed.", exc_info=True)

    # ── Valid DAG % shift figure ──────────────────────────────────
    # This figure loads DiBS .pt.gz inference artifacts to compute
    # per-sample DAG validity, so it is gated by --skip-posterior.
    if skip_posterior:
        log.info("Valid DAG shift figure skipped (--skip-posterior).")
    else:
        try:
            valid_dag_df = gen_plots.generate_valid_dag_shift_figure(
                raw_df,
                run_dirs,
                output_path=results_dir / "valid_dag_shift.pdf",
            )
        except Exception:
            if strict:
                raise
            log.warning("Valid DAG shift figure failed.", exc_info=True)

    # ── Error decomposition (FP/FN/Reversed) table ────────────────
    try:
        gen_plots.generate_error_decomposition_table(
            raw_df,
            output_path=results_dir / "error_decomposition.tex",
        )
    except Exception:
        if strict:
            raise
        log.warning("Error decomposition figure failed.", exc_info=True)

    # ── Cross-axis summary table (ne-SID, E-F1, ne-SHD) ─────────
    try:
        gen_plots.generate_cross_axis_summary_table(
            raw_df,
            output_path=results_dir / "cross_axis_summary.tex",
        )
    except Exception:
        if strict:
            raise
        log.warning("Cross-axis summary table failed.", exc_info=True)

    for score_metric in ("edge_entropy", "graph_nll_per_edge"):
        fig_name = f"uncertainty_scatter_{score_metric}.pdf"
        try:
            unc_plots.generate_uncertainty_scatter(
                raw_df,
                score_metric=score_metric,
                output_path=results_dir / fig_name,
            )
        except EmptyAnalysisDataError:
            if strict:
                raise
            log.warning(
                "Uncertainty scatter for '%s' skipped (no data).",
                score_metric,
            )

    for axis, stem in (("nodes", "node_transfer"), ("samples", "sample_transfer")):
        generate_transfer_figure(
            raw_df,
            axis=axis,
            output_path=results_dir / f"{stem}.pdf",
        )

    unc_plots.generate_ood_detection_summary_table(
        raw_df, results_dir / "ood_detection.tex"
    )

    unc_plots.generate_ece_summary_table(raw_df, results_dir / "ece_summary.tex")

    gen_tables.generate_robustness_table(
        summary_df, appendix_e_dir / "fixed_ood_appendix.tex"
    )

    try:
        gen_tables.generate_distance_regression_table(
            summary_df, results_dir / "distance_regression.tex"
        )
    except Exception:
        if strict:
            raise
        log.warning("Distance regression table generation failed.", exc_info=True)

    if skip_posterior:
        log.info("Posterior diagnostics skipped (--skip-posterior).")
    else:
        try:
            posterior_df = diag_posterior.run_posterior_diagnostics_from_runs(run_dirs)
            if not posterior_df.empty:
                posterior_df = posterior_df.copy()
                posterior_df["Model"] = posterior_df["Model"].map(
                    thesis_common.paper_model_label
                )
                generate_event_probability_bar(
                    posterior_df, results_dir / "event_probabilities.pdf"
                )
                generate_posterior_diagnostic_violins(
                    posterior_df, results_dir / "posterior_diagnostics.pdf"
                )
        except Exception:
            if strict:
                raise
            log.warning("Posterior diagnostics failed.", exc_info=True)

    thesis_common.write_results_macros(
        selected_runs, results_dir / "results_macros.tex"
    )

    generate_appendix_artifacts(thesis_root, thesis_common.CONFIGS_ROOT)

    log.info(
        "Generated thesis artifacts under %s", thesis_root / thesis_common.GRAPHICS_ROOT
    )
    return thesis_root / thesis_common.GRAPHICS_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild paper/final_thesis/graphics/ from curated runs under "
            "experiments/thesis_runs/."
        )
    )
    parser.add_argument(
        "--input-root",
        type=str,
        default="experiments/thesis_runs",
        help="Curated one-run-per-model input root (default: experiments/thesis_runs).",
    )
    parser.add_argument(
        "--thesis-root",
        type=str,
        default="paper/final_thesis",
        help="Thesis repository root containing graphics/ chapter directories.",
    )
    parser.add_argument(
        "--best-effort",
        action="store_true",
        help="Continue when optional diagnostics fail instead of aborting.",
    )
    parser.add_argument(
        "--skip-posterior",
        action="store_true",
        help="Skip posterior diagnostics (avoids loading .pt.gz inference artifacts).",
    )
    args = parser.parse_args()

    run_thesis_analysis(
        input_root=Path(args.input_root),
        thesis_root=Path(args.thesis_root),
        strict=not bool(args.best_effort),
        skip_posterior=bool(args.skip_posterior),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except (
        EmptyAnalysisDataError,
        FileNotFoundError,
        thesis_common.ThesisRunSelectionError,
    ) as exc:
        log.error("Thesis analysis failed: %s", exc)
        raise SystemExit(1) from exc
