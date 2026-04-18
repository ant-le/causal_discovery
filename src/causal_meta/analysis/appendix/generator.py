from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from omegaconf import OmegaConf


def _escape_tex(value: str) -> str:
    for char, repl in (
        ("\\", r"\textbackslash{}"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
    ):
        value = value.replace(char, repl)
    return value


def _load_yaml(path: Path) -> dict[str, Any]:
    data = OmegaConf.to_container(OmegaConf.load(path), resolve=False)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping at {path}")
    return data


def _fmt(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(_fmt(v) for v in value)
    if isinstance(value, dict):
        parts = [f"{k}={_fmt(v)}" for k, v in value.items()]
        return "; ".join(parts)
    return str(value)


def _graph_summary(cfg: Mapping[str, Any]) -> str:
    graph_type = str(cfg.get("type", ""))
    if graph_type == "er":
        return f"ER ({cfg.get('sparsity')})"
    if graph_type == "sf":
        return f"SF (m={cfg.get('m')})"
    if graph_type == "sbm":
        return f"SBM (b={cfg.get('n_blocks')}, in={cfg.get('p_intra')}, out={cfg.get('p_inter')})"
    if graph_type == "mixture":
        generators = cfg.get("generators", [])
        return f"Mixture ({len(generators)} generators)"
    return _fmt(cfg)


def _mech_summary(cfg: Mapping[str, Any]) -> str:
    mech_type = str(cfg.get("type", ""))
    pretty = mech_type.replace("_", " ").title()
    if mech_type == "mlp":
        return f"MLP (h={cfg.get('hidden_dim')})"
    if mech_type == "gp":
        return f"GP ({cfg.get('mode')}, rff={cfg.get('rff_dim')})"
    if mech_type == "pnl":
        return f"PNL ({cfg.get('nonlinearity_type')})"
    if mech_type == "mixture":
        factories = cfg.get("factories", [])
        return f"Mixture ({len(factories)} factories)"
    return pretty if mech_type else _fmt(cfg)


def _write_table(
    output_path: Path,
    *,
    caption: str,
    label: str,
    headers: list[str],
    rows: list[list[str]],
    colspec: str,
    resize: bool = False,
    font_size: str = r"\footnotesize",
) -> None:
    lines = [r"\begin{table}[h]", r"\centering"]
    if font_size:
        lines.append(font_size)
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    if resize:
        lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(rf"\begin{{tabular}}{{{colspec}}}")
    lines.append(r"\toprule")
    header_line = " & ".join(rf"\textbf{{{_escape_tex(h)}}}" for h in headers) + r" \\"
    lines.append(header_line)
    lines.append(r"\midrule")
    for row in rows:
        lines.append(" & ".join(_escape_tex(cell) for cell in row) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    if resize:
        lines.append(r"}")
    lines.append(r"\end{table}")
    output_path.write_text("\n".join(lines) + "\n")


def _write_family_table(
    output_path: Path,
    *,
    caption: str,
    label: str,
    rows: list[list[str]],
) -> None:
    _write_table(
        output_path,
        caption=caption,
        label=label,
        headers=["Family", "Category", "Graph", "Mechanism", "N_G", "N"],
        rows=rows,
        colspec=(
            ">{\\raggedright\\arraybackslash}p{3.2cm}"
            ">{\\raggedright\\arraybackslash}p{1.8cm}"
            ">{\\raggedright\\arraybackslash}p{2.5cm}"
            ">{\\raggedright\\arraybackslash}p{2.3cm}cc"
        ),
        resize=True,
        font_size=r"\scriptsize",
    )


def _family_category(name: str) -> str:
    if name.startswith("id_"):
        return "ID"
    if name.startswith("ood_graph_"):
        return "Graph-OOD"
    if name.startswith("ood_mech_"):
        return "Mechanism-OOD"
    if name.startswith("ood_noise_"):
        return "Noise-OOD"
    if name.startswith("ood_both_"):
        return "Compound-OOD"
    if name.startswith("ood_nodes_"):
        return "Node-count"
    if name.startswith("ood_samples_"):
        return "Sample-count"
    return "Other"


def _family_label(name: str) -> str:
    if name.startswith("id_"):
        if "linear" in name and "er20" in name:
            return "ID: Linear ER20"
        if "neuralnet" in name and "sf2" in name:
            return "ID: NN SF2"
        if "gpcde" in name and "er60" in name:
            return "ID: GP ER60"
    if name.startswith("ood_graph_"):
        if "linear" in name:
            return "Graph: SBM + Linear"
        if "neuralnet" in name:
            return "Graph: SBM + NN"
        if "gpcde" in name:
            return "Graph: SBM + GP"
    if name.startswith("ood_mech_"):
        if "periodic" in name:
            return "Mech: Periodic"
        if "square" in name:
            return "Mech: Square"
        if "logistic_map" in name:
            return "Mech: Logistic map"
        if "pnl_tanh" in name:
            return "Mech: PNL tanh"
    if name.startswith("ood_noise_"):
        if "laplace" in name:
            return "Noise: Laplace"
        if "uniform" in name:
            return "Noise: Uniform"
    if name.startswith("ood_both_"):
        # Extract graph type: ood_both_<graph>_<mech>_d<nodes>_n<samples>
        graph_codes = {"sbm": "SBM", "ws": "WS", "grg": "GRG"}
        graph_label = "SBM"
        for code, label in graph_codes.items():
            if f"ood_both_{code}_" in name:
                graph_label = label
                break
        mech_codes = {
            "periodic": "Periodic",
            "logistic_map": "Logistic map",
            "pnl_tanh": "PNL",
        }
        for code, mech_label in mech_codes.items():
            if code in name:
                suffix = ""
                if "d60" in name:
                    # Include node/sample info for large-graph variants
                    d = name.split("_d")[-1].split("_")[0]
                    n = name.split("_n")[-1]
                    suffix = f" ({d},{n})"
                return f"Compound: {graph_label} + {mech_label}{suffix}"
    if name.startswith("ood_nodes_"):
        if "neuralnet" in name:
            return f"Nodes: NN SF2 ({name.split('_d')[-1].split('_')[0]})"
        return f"Nodes: Linear SF2 ({name.split('_d')[-1].split('_')[0]})"
    if name.startswith("ood_samples_"):
        n_samples = name.split("_n")[-1]
        if "neuralnet" in name:
            return f"Samples: NN SF2 ({n_samples})"
        return f"Samples: Linear ER20 ({n_samples})"
    return name


def _short_group_prefixes(value: Mapping[str, Any]) -> str:
    keys = [str(k) for k in value.keys()]
    return ", ".join(keys)


def _hydra_template_summary(value: Any) -> str:
    text = _fmt(value)
    if "CAUSAL_META_RUN_DIR" in text and "model.id" in text:
        return "env-driven run dir / model"
    if "CAUSAL_META_RUN_DIR" in text:
        return "env-driven run dir"
    if "CAUSAL_META_SWEEP_DIR" in text:
        return "env-driven sweep dir"
    if "${model.id}" in text:
        return "model.id"
    if "${.n_samples}" in text:
        return "inherits n_samples"
    return text


def generate_appendix_artifacts(thesis_root: Path, configs_root: Path) -> list[str]:
    from causal_meta.analysis.common.thesis import (
        GRAPHICS_APPENDIX_A,
        GRAPHICS_APPENDIX_B,
        GRAPHICS_APPENDIX_C,
        GRAPHICS_APPENDIX_F,
    )

    dir_a = thesis_root / GRAPHICS_APPENDIX_A
    dir_b = thesis_root / GRAPHICS_APPENDIX_B
    dir_c = thesis_root / GRAPHICS_APPENDIX_C
    dir_f = thesis_root / GRAPHICS_APPENDIX_F
    for d in (dir_a, dir_b, dir_c, dir_f):
        d.mkdir(parents=True, exist_ok=True)

    default_cfg = _load_yaml(configs_root / "default.yaml")
    benchmark_cfg = _load_yaml(configs_root / "dg_2pretrain_multimodel.yaml")
    smoke_cfg = _load_yaml(configs_root / "dg_2pretrain_smoke.yaml")
    trainer_cfg = _load_yaml(configs_root / "trainer" / "default.yaml")
    inference_cfg = _load_yaml(configs_root / "inference" / "default.yaml")
    avici_cfg = _load_yaml(configs_root / "model" / "avici.yaml")
    bcnp_cfg = _load_yaml(configs_root / "model" / "bcnp.yaml")
    dibs_cfg = _load_yaml(configs_root / "model" / "dibs.yaml")
    bayesdag_cfg = _load_yaml(configs_root / "model" / "bayesdag.yaml")
    random_cfg = _load_yaml(configs_root / "model" / "random.yaml")

    generated: list[str] = []

    exp_rows = [
        ["Benchmark config", "dg_2pretrain_multimodel.yaml"],
        ["Smoke config", smoke_cfg.get("name", "dg_2pretrain_smoke")],
        ["Train node counts", _fmt(benchmark_cfg["data"].get("train_n_nodes", []))],
        [
            "Global samples per task",
            _fmt(benchmark_cfg["data"].get("samples_per_task")),
        ],
        ["Validation seeds", str(len(benchmark_cfg["data"].get("seeds_val", [])))],
        ["Test seeds", str(len(benchmark_cfg["data"].get("seeds_test", [])))],
        ["Base seed", _fmt(benchmark_cfg["data"].get("base_seed"))],
        ["Batch size (train)", _fmt(benchmark_cfg["data"].get("batch_size_train"))],
        [
            "Batch size (val/test)",
            f"{benchmark_cfg['data'].get('batch_size_val')} / {benchmark_cfg['data'].get('batch_size_test')}",
        ],
        [
            "Workers / pin memory",
            f"{benchmark_cfg['data'].get('num_workers')} / {benchmark_cfg['data'].get('pin_memory')}",
        ],
        ["Max tasks seen", _fmt(trainer_cfg.get("max_tasks_seen"))],
        [
            "Validation interval (tasks)",
            _fmt(trainer_cfg.get("val_check_interval_tasks")),
        ],
        [
            "Checkpoint interval (tasks)",
            _fmt(trainer_cfg.get("checkpoint_every_n_tasks")),
        ],
        ["Learning rate", _fmt(trainer_cfg.get("lr"))],
        ["Weight decay", _fmt(trainer_cfg.get("weight_decay"))],
        [
            "Gradient accumulation",
            _fmt(
                benchmark_cfg.get("trainer", {}).get(
                    "accumulate_grad_batches",
                    trainer_cfg.get("accumulate_grad_batches"),
                )
            ),
        ],
        ["AMP / dtype", f"{trainer_cfg.get('amp')} / {trainer_cfg.get('amp_dtype')}"],
        [
            "Posterior samples",
            _fmt(
                benchmark_cfg.get("inference", {}).get(
                    "n_samples", inference_cfg.get("n_samples")
                )
            ),
        ],
        ["Cache compression", _fmt(inference_cfg.get("cache_compress"))],
    ]
    _write_table(
        dir_a / "experimental_settings_table.tex",
        caption="Global experimental settings derived from the benchmark Hydra configurations.",
        label="tab:appendix_experimental_settings",
        headers=["Setting", "Value"],
        rows=exp_rows,
        colspec="lp{9cm}",
    )
    generated.append(f"{GRAPHICS_APPENDIX_A}/experimental_settings_table.tex")

    eval_rows = [
        ["Validation sample count", _fmt(trainer_cfg.get("validation_n_samples"))],
        [
            "Validation metrics",
            _fmt(
                benchmark_cfg.get("trainer", {}).get(
                    "validation_metrics", trainer_cfg.get("validation_metrics", [])
                )
            ),
        ],
        [
            "Validation selection metric",
            _fmt(
                benchmark_cfg.get("trainer", {}).get(
                    "validation_selection_metric",
                    trainer_cfg.get("validation_selection_metric"),
                )
            ),
        ],
        [
            "Validation group prefixes",
            _short_group_prefixes(trainer_cfg.get("validation_group_prefixes", {})),
        ],
        ["AUC shuffles", _fmt(inference_cfg.get("auc_num_shuffles"))],
        ["Class-balanced AUC", _fmt(inference_cfg.get("auc_balance_classes"))],
        ["AUC seed", _fmt(inference_cfg.get("auc_seed"))],
    ]
    _write_table(
        dir_a / "evaluation_settings_table.tex",
        caption="Evaluation and validation settings used by the thesis analysis pipeline.",
        label="tab:appendix_evaluation_settings",
        headers=["Setting", "Value"],
        rows=eval_rows,
        colspec="lp{9cm}",
    )
    generated.append(f"{GRAPHICS_APPENDIX_A}/evaluation_settings_table.tex")

    model_rows = [
        [
            "AviCi",
            f"dim={avici_cfg.get('dim')}; layers={avici_cfg.get('layers')}; heads={avici_cfg.get('num_heads')}; dropout={avici_cfg.get('dropout')}",
        ],
        [
            "BCNP",
            f"d_model={bcnp_cfg.get('d_model')}; layers={bcnp_cfg.get('num_layers')}; decoder_layers={bcnp_cfg.get('num_layers_decoder')}; sinkhorn_iter={bcnp_cfg.get('sinkhorn_iter')}",
        ],
        [
            "DiBS",
            f"mode={dibs_cfg.get('mode')}; steps={dibs_cfg.get('steps')}; particles={dibs_cfg.get('n_particles')}; timeout={dibs_cfg.get('external_timeout_s')}s",
        ],
        [
            "BayesDAG",
            f"variant={bayesdag_cfg.get('variant')}; chains={bayesdag_cfg.get('num_chains')}; max_epochs={bayesdag_cfg.get('max_epochs')}; timeout={bayesdag_cfg.get('external_timeout_s')}s",
        ],
        [
            "Random",
            f"randomize_topological_order={random_cfg.get('randomize_topological_order')}",
        ],
    ]
    _write_table(
        dir_b / "model_configurations_table.tex",
        caption="Model-specific configurations derived from the Hydra model YAML files.",
        label="tab:appendix_model_configurations",
        headers=["Model", "Configuration summary"],
        rows=model_rows,
        colspec="lp{10cm}",
    )
    generated.append(f"{GRAPHICS_APPENDIX_B}/model_configurations_table.tex")

    source_rows = [
        [
            _fmt(benchmark_cfg["data"].get("train_n_nodes", [])),
            _graph_summary(benchmark_cfg["data"]["train_family"]["graph_cfg"]),
            _mech_summary(benchmark_cfg["data"]["train_family"]["mech_cfg"]),
            _fmt(benchmark_cfg["data"]["samples_per_task"]),
        ]
    ]
    _write_table(
        dir_c / "source_distribution_table.tex",
        caption="Shared source distribution used for amortized pre-training.",
        label="tab:appendix_source_distribution",
        headers=[
            "Train node counts",
            "Graph support",
            "Mechanism support",
            "Samples per task",
        ],
        rows=source_rows,
        colspec="p{2.4cm}p{4.0cm}p{4.0cm}p{2.2cm}",
        resize=True,
    )
    generated.append(f"{GRAPHICS_APPENDIX_C}/source_distribution_table.tex")

    val_rows: list[list[str]] = []
    for name, cfg in benchmark_cfg["data"].get("val_families", {}).items():
        val_rows.append(
            [
                _family_label(str(name)),
                _family_category(str(name)),
                _graph_summary(cfg.get("graph_cfg", {})),
                _mech_summary(cfg.get("mech_cfg", {})),
                _fmt(cfg.get("n_nodes")),
                _fmt(cfg.get("samples_per_task")),
            ]
        )
    _write_family_table(
        dir_c / "validation_families_table.tex",
        caption="Validation families used during benchmark training and selection.",
        label="tab:appendix_validation_families",
        rows=val_rows,
    )
    generated.append(f"{GRAPHICS_APPENDIX_C}/validation_families_table.tex")

    test_rows: list[list[str]] = []
    for name, cfg in benchmark_cfg["data"].get("test_families", {}).items():
        test_rows.append(
            [
                _family_label(str(name)),
                _family_category(str(name)),
                _graph_summary(cfg.get("graph_cfg", {})),
                _mech_summary(cfg.get("mech_cfg", {})),
                _fmt(cfg.get("n_nodes")),
                _fmt(cfg.get("samples_per_task")),
            ]
        )
    fixed_rows = [
        row
        for row in test_rows
        if row[1] in {"ID", "Graph-OOD", "Mechanism-OOD", "Noise-OOD", "Compound-OOD"}
    ]
    transfer_rows = [
        row for row in test_rows if row[1] in {"Node-count", "Sample-count"}
    ]
    _write_family_table(
        dir_c / "test_families_fixed_table.tex",
        caption="Fixed-size test families used in the in-distribution and OOD benchmark suites.",
        label="tab:appendix_test_families_fixed",
        rows=fixed_rows,
    )
    _write_family_table(
        dir_c / "test_families_transfer_table.tex",
        caption="Task-regime transfer families used for node-count and sample-count evaluation.",
        label="tab:appendix_test_families_transfer",
        rows=transfer_rows,
    )
    generated.extend(
        [
            f"{GRAPHICS_APPENDIX_C}/test_families_fixed_table.tex",
            f"{GRAPHICS_APPENDIX_C}/test_families_transfer_table.tex",
        ]
    )

    hydra_cfg = benchmark_cfg.get("hydra", {})
    reproducibility_rows = [
        ["Hydra run dir", _hydra_template_summary(hydra_cfg.get("run", {}).get("dir"))],
        [
            "Hydra sweep dir",
            _hydra_template_summary(hydra_cfg.get("sweep", {}).get("dir")),
        ],
        [
            "Hydra sweep subdir",
            _hydra_template_summary(hydra_cfg.get("sweep", {}).get("subdir")),
        ],
        [
            "Hydra output_subdir",
            _hydra_template_summary(hydra_cfg.get("output_subdir")),
        ],
        [
            "Default installed run dir",
            _hydra_template_summary(
                default_cfg.get("hydra", {}).get("run", {}).get("dir")
            ),
        ],
        ["Cache compression", _fmt(inference_cfg.get("cache_compress"))],
        ["Cache dtype", _fmt(inference_cfg.get("cache_dtype"))],
        [
            "Cache sample count",
            _hydra_template_summary(inference_cfg.get("cache_n_samples")),
        ],
        ["Safety checks", _fmt(benchmark_cfg.get("data", {}).get("safety_checks"))],
        ["Normalize data", _fmt(benchmark_cfg.get("data", {}).get("normalize_data"))],
    ]
    _write_table(
        dir_f / "reproducibility_table.tex",
        caption="Hydra output conventions and reproducibility-relevant artifact settings.",
        label="tab:appendix_reproducibility",
        headers=["Setting", "Value"],
        rows=reproducibility_rows,
        colspec="lp{9cm}",
    )
    generated.append(f"{GRAPHICS_APPENDIX_F}/reproducibility_table.tex")

    return generated
