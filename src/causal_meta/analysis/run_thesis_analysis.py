from __future__ import annotations

import argparse
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

from causal_meta.analysis.appendix.generator import generate_appendix_artifacts
from causal_meta.analysis.common import thesis as thesis_common
from causal_meta.analysis.plots.results import (
    generate_event_probability_bar,
    generate_failure_mode_bar,
    generate_per_model_failure_mode_bar,
    generate_posterior_diagnostic_violins,
    generate_selective_prediction_pareto,
)
from causal_meta.analysis.rq1 import failure_modes as rq1_failure_modes
from causal_meta.analysis.rq1 import plots as rq1_plots
from causal_meta.analysis.rq1 import tables as rq1_tables
from causal_meta.analysis.rq2.transfer import generate_transfer_figure
from causal_meta.analysis.rq3 import diagnostics as rq3_diagnostics
from causal_meta.analysis.rq3 import ood_detection as rq3_ood
from causal_meta.analysis.rq3 import plots as rq3_plots
from causal_meta.analysis.utils import AMORTISED_MODELS, EmptyAnalysisDataError

log = logging.getLogger(__name__)


def run_thesis_analysis(
    *,
    input_root: Path,
    thesis_root: Path,
    strict: bool = True,
    skip_posterior: bool = False,
) -> Path:
    """Run the curated thesis analysis pipeline and rebuild generated outputs."""

    selected_runs = thesis_common.resolve_thesis_run_directories(input_root)
    run_dirs = [run.run_dir for run in selected_runs]
    temp_root = thesis_common.prepare_generated_workspace(thesis_root)
    generated_files: list[str] = []

    try:
        summary_df = thesis_common.prepare_summary_dataframe(run_dirs)
        raw_df = thesis_common.prepare_raw_dataframe(run_dirs)

        data_dir = temp_root / "data"
        tables_dir = temp_root / "tables"
        figures_dir = temp_root / "figures"
        snippets_dir = temp_root / "snippets"
        provenance_dir = temp_root / "provenance"

        summary_df.to_csv(data_dir / "summary_metrics.csv", index=False)
        raw_df.to_csv(data_dir / "raw_task_metrics.csv", index=False)
        generated_files.extend(
            [
                "data/summary_metrics.csv",
                "data/raw_task_metrics.csv",
            ]
        )

        anchor_df = rq1_plots.generate_results_anchor_table(
            raw_df, tables_dir / "results_anchor.tex"
        )
        anchor_df.to_csv(data_dir / "results_anchor.csv", index=False)
        generated_files.extend(
            [
                "tables/results_anchor.tex",
                "data/results_anchor.csv",
            ]
        )

        for shift_key in ("graph", "mechanism", "noise", "compound"):
            fig_name = f"shift_{shift_key}.pdf"
            csv_name = f"shift_{shift_key}.csv"
            try:
                shift_df = rq1_plots.generate_shift_figure(
                    raw_df,
                    shift_axis=shift_key,
                    output_path=figures_dir / fig_name,
                )
                shift_df.to_csv(data_dir / csv_name, index=False)
                generated_files.extend([f"figures/{fig_name}", f"data/{csv_name}"])
            except EmptyAnalysisDataError:
                if strict:
                    raise
                log.warning("Shift figure for '%s' skipped (no data).", shift_key)

        for score_metric in ("edge_entropy", "graph_nll_per_edge"):
            fig_name = f"uncertainty_scatter_{score_metric}.pdf"
            csv_name = f"uncertainty_scatter_{score_metric}.csv"
            try:
                scatter_df = rq3_plots.generate_uncertainty_scatter(
                    raw_df,
                    score_metric=score_metric,
                    output_path=figures_dir / fig_name,
                )
                scatter_df.to_csv(data_dir / csv_name, index=False)
                generated_files.extend([f"figures/{fig_name}", f"data/{csv_name}"])
            except EmptyAnalysisDataError:
                if strict:
                    raise
                log.warning(
                    "Uncertainty scatter for '%s' skipped (no data).",
                    score_metric,
                )

        for axis, stem in (("nodes", "node_transfer"), ("samples", "sample_transfer")):
            transfer_df = generate_transfer_figure(
                raw_df,
                axis=axis,
                output_path=figures_dir / f"{stem}.pdf",
            )
            transfer_df.to_csv(data_dir / f"{stem}.csv", index=False)
            generated_files.extend([f"figures/{stem}.pdf", f"data/{stem}.csv"])

        ood_detection_df = rq3_plots.generate_ood_detection_summary_table(
            raw_df, tables_dir / "ood_detection.tex"
        )
        ood_detection_df.to_csv(data_dir / "ood_detection.csv", index=False)
        generated_files.extend(
            [
                "tables/ood_detection.tex",
                "data/ood_detection.csv",
            ]
        )

        ece_df = rq3_plots.generate_ece_summary_table(
            raw_df, tables_dir / "ece_summary.tex"
        )
        ece_df.to_csv(data_dir / "ece_summary.csv", index=False)
        generated_files.extend(
            [
                "tables/ece_summary.tex",
                "data/ece_summary.csv",
            ]
        )

        selective_df = rq3_ood.compute_selective_prediction(raw_df)
        if not selective_df.empty:
            generate_selective_prediction_pareto(
                selective_df, figures_dir / "selective_prediction.pdf"
            )
            selective_df.to_csv(data_dir / "selective_prediction.csv", index=False)
            generated_files.extend(
                [
                    "figures/selective_prediction.pdf",
                    "data/selective_prediction.csv",
                ]
            )

        rq1_tables.generate_robustness_table(
            summary_df, tables_dir / "fixed_ood_appendix.tex"
        )
        generated_files.append("tables/fixed_ood_appendix.tex")

        try:
            rq1_tables.generate_distance_regression_table(
                summary_df, tables_dir / "distance_regression.tex"
            )
            generated_files.append("tables/distance_regression.tex")
        except Exception:
            if strict:
                raise
            log.warning("Distance regression table generation failed.", exc_info=True)

        try:
            failure_df = raw_df[
                raw_df["Metric"].isin(
                    ["sparsity_ratio", "skeleton_f1", "orientation_accuracy"]
                )
            ].copy()
            classified = rq1_failure_modes.classify_failure_modes(failure_df)
            if not classified.empty:
                fractions = rq1_failure_modes.failure_mode_fractions(classified)
                generate_failure_mode_bar(fractions, figures_dir / "failure_modes.pdf")
                fractions.to_csv(data_dir / "failure_modes.csv", index=False)
                generated_files.extend(
                    [
                        "figures/failure_modes.pdf",
                        "data/failure_modes.csv",
                    ]
                )
                for model_name in sorted(AMORTISED_MODELS):
                    safe_name = model_name.lower().replace("-", "_")
                    fig_name = f"failure_modes_{safe_name}.pdf"
                    generate_per_model_failure_mode_bar(
                        fractions,
                        model=model_name,
                        output_path=figures_dir / fig_name,
                    )
                    generated_files.append(f"figures/{fig_name}")
        except Exception:
            if strict:
                raise
            log.warning("Failure-mode analysis failed.", exc_info=True)

        if skip_posterior:
            log.info("Posterior diagnostics skipped (--skip-posterior).")
        else:
            try:
                posterior_df = rq3_diagnostics.run_posterior_diagnostics_from_runs(
                    run_dirs
                )
                if not posterior_df.empty:
                    posterior_df = posterior_df.copy()
                    posterior_df["Model"] = posterior_df["Model"].map(
                        thesis_common.paper_model_label
                    )
                    generate_event_probability_bar(
                        posterior_df, figures_dir / "event_probabilities.pdf"
                    )
                    generate_posterior_diagnostic_violins(
                        posterior_df, figures_dir / "posterior_diagnostics.pdf"
                    )
                    generated_files.extend(
                        [
                            "figures/event_probabilities.pdf",
                            "figures/posterior_diagnostics.pdf",
                        ]
                    )
            except Exception:
                if strict:
                    raise
                log.warning("Posterior diagnostics failed.", exc_info=True)

        thesis_common.write_selected_runs(
            selected_runs, provenance_dir / "selected_runs.json"
        )
        thesis_common.write_results_macros(
            selected_runs, snippets_dir / "results_macros.tex"
        )
        generated_files.extend(
            [
                "provenance/selected_runs.json",
                "snippets/results_macros.tex",
            ]
        )

        appendix_generated = generate_appendix_artifacts(
            temp_root / "appendix", thesis_common.CONFIGS_ROOT
        )
        generated_files.extend(appendix_generated)

        has_mock_runs = any(
            "mock" in run.run_name.lower() or "mock" in run.run_id.lower()
            for run in selected_runs
        )
        if has_mock_runs:
            log.warning(
                "Selected thesis runs include mock/provisional outputs; treat generated artifacts as provisional."
            )

        analysis_report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "input_root": str(input_root.resolve()),
            "thesis_root": str(thesis_root.resolve()),
            "strict": bool(strict),
            "has_mock_runs": has_mock_runs,
            "generated_files": sorted(generated_files),
        }
        thesis_common.write_json(
            provenance_dir / "analysis_report.json", analysis_report
        )
        generated_files.append("provenance/analysis_report.json")

        final_root = thesis_common.finalize_generated_workspace(temp_root, thesis_root)
        log.info("Generated thesis artifacts under %s", final_root)
        return final_root
    except Exception:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild paper/final_thesis/generated from curated runs under "
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
        help="Thesis repository root containing the generated/ folder.",
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
