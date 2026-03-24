from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Mapping, cast

import torch
import torch.nn as nn
from omegaconf import DictConfig

from causal_meta.datasets.data_module import CausalMetaModule
from causal_meta.datasets.generators.configs import FamilyConfig
from causal_meta.datasets.torch_datasets import MetaFixedDataset
from causal_meta.datasets.utils.normalization import normalize_scm_data
from causal_meta.runners.logger.base import BaseLogger
from causal_meta.runners.metrics.graph import Metrics
from causal_meta.runners.metrics.scm import SCMMetrics
from causal_meta.runners.tasks.utils import (
    infer_device,
    resolve_inference_root,
    sampling_mode,
    shard_indices,
    unwrap_model,
)
from causal_meta.runners.utils.artifacts import (
    NpEncoder,
    atomic_torch_save,
    cache_settings,
    cache_suffix,
    find_inference_artifact,
    get_model_name,
    prepare_graph_samples_for_cache,
    resolve_output_dir,
    torch_load,
)
from causal_meta.runners.utils.distributed import DistributedContext
from causal_meta.runners.utils.explicit_profiles import (
    apply_explicit_profile,
    infer_explicit_profile,
)

log = logging.getLogger(__name__)


def _prepare_amortized_model_input(
    batch: Mapping[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Prepare a model input tensor and optional node mask."""
    input_data = batch["data"].to(device)
    intervention_mask = batch.get("intervention_mask")
    if intervention_mask is not None:
        input_data = torch.stack([input_data, intervention_mask.to(device)], dim=-1)

    node_mask = batch.get("node_mask")
    if node_mask is not None:
        node_mask = node_mask.to(device)

    return input_data, node_mask


def _extract_graph_type(family_cfg: FamilyConfig) -> str:
    """Infer the graph type string from a FamilyConfig's graph_cfg."""
    graph_cfg = family_cfg.graph_cfg
    type_attr = getattr(graph_cfg, "type", None)
    if type_attr is not None:
        return str(type_attr)
    # Fall back to class name heuristic
    cls_name = type(graph_cfg).__name__.lower()
    for tag in ("er", "scalefree", "sf", "sbm", "mixture"):
        if tag in cls_name:
            # Normalize "scalefree" to "sf"
            return "sf" if tag == "scalefree" else tag
    return "unknown"


def _extract_mech_type(family_cfg: FamilyConfig) -> str:
    """Infer the mechanism type string from a FamilyConfig's mech_cfg."""
    mech_cfg = family_cfg.mech_cfg
    type_attr = getattr(mech_cfg, "type", None)
    if type_attr is not None:
        return str(type_attr)
    cls_name = type(mech_cfg).__name__.lower()
    for tag in (
        "linear",
        "mlp",
        "gp",
        "periodic",
        "square",
        "logistic",
        "pnl",
        "mixture",
    ):
        if tag in cls_name:
            return tag
    return "unknown"


def _extract_sparsity_param(family_cfg: FamilyConfig) -> float | None:
    """Extract the sparsity-related parameter from a graph config, if available."""
    graph_cfg = family_cfg.graph_cfg
    # ER: sparsity or edge_prob
    sparsity = getattr(graph_cfg, "sparsity", None)
    if sparsity is not None:
        return float(sparsity)
    edge_prob = getattr(graph_cfg, "edge_prob", None)
    if edge_prob is not None:
        return float(edge_prob)
    # SF: attachment parameter m
    m = getattr(graph_cfg, "m", None)
    if m is not None:
        return float(m)
    return None


def _build_family_metadata(
    test_family_cfgs: dict[str, FamilyConfig],
) -> dict[str, dict[str, Any]]:
    """Build a dataset_key -> {n_nodes, graph_type, mech_type, sparsity_param} map."""
    result: dict[str, dict[str, Any]] = {}
    for name, fcfg in test_family_cfgs.items():
        entry: dict[str, Any] = {
            "n_nodes": fcfg.n_nodes,
            "graph_type": _extract_graph_type(fcfg),
            "mech_type": _extract_mech_type(fcfg),
        }
        sp = _extract_sparsity_param(fcfg)
        if sp is not None:
            entry["sparsity_param"] = sp
        result[name] = entry
    return result


def _prepare_cached_samples_for_metrics(
    artifact: Mapping[str, Any],
    *,
    requested_n_samples: int,
) -> tuple[torch.Tensor | None, int | None]:
    """Validate and reshape cached graph samples for metric computation.

    Args:
        artifact: Loaded artifact payload.
        requested_n_samples: Number of posterior samples requested by config.

    Returns:
        Tuple ``(samples_for_metrics, cached_n_samples)`` where:
        - ``samples_for_metrics`` has shape ``(S, 1, N, N)`` when compatible,
          otherwise ``None``.
        - ``cached_n_samples`` is the number of samples found in cache, when
          extractable.
    """
    graph_samples = artifact.get("graph_samples")
    if not isinstance(graph_samples, torch.Tensor):
        return None, None

    # Cached explicit-model artifacts are written as (B, K, N, N) with B=1.
    if graph_samples.ndim == 4:
        if int(graph_samples.shape[0]) != 1:
            return None, int(graph_samples.shape[1])
        cached_n_samples = int(graph_samples.shape[1])
        if cached_n_samples < requested_n_samples:
            return None, cached_n_samples
        prepared = graph_samples[:, :requested_n_samples].permute(1, 0, 2, 3)
        return prepared, cached_n_samples

    # Backward-compatible fallback for artifacts saved as (K, N, N).
    if graph_samples.ndim == 3:
        cached_n_samples = int(graph_samples.shape[0])
        if cached_n_samples < requested_n_samples:
            return None, cached_n_samples
        prepared = graph_samples[:requested_n_samples].unsqueeze(1)
        return prepared, cached_n_samples

    return None, None


def run(
    cfg: DictConfig,
    model: nn.Module,
    data_module: CausalMetaModule,
    *,
    logger: BaseLogger | None = None,
    output_dir: Path | None = None,
) -> None:
    # DDP Setup
    dist_ctx = DistributedContext.current()
    rank = dist_ctx.rank

    if rank == 0:
        log.info(f"Starting evaluation: {cfg.name}")

    # Device (explicit models may have no parameters)
    device = infer_device(model, dist_ctx)

    # Data
    test_loaders = data_module.test_dataloader()

    # Pre-compute family metadata; distances are pre-computed by data_module.setup()
    dm_config = getattr(data_module, "config", None)
    test_family_cfgs = getattr(dm_config, "test_families", None) or {}
    family_metadata = _build_family_metadata(test_family_cfgs)
    family_distances = getattr(data_module, "family_distances", {}) or {}

    model.eval()

    # Unwrap if DDP
    model_unwrapped = unwrap_model(model)
    model_unwrapped_raw = model.module if dist_ctx.is_distributed else model

    all_summary_metrics = {}
    all_raw_metrics = {}

    n_samples = int(cfg.inference.get("n_samples", 100))
    auc_num_shuffles = int(cfg.inference.get("auc_num_shuffles", 1000))
    auc_balance_classes = bool(cfg.inference.get("auc_balance_classes", True))
    auc_seed = int(cfg.inference.get("auc_seed", 0))

    # Initialize Metrics Handlers (default list includes all graph metrics)
    metrics_handler = Metrics(
        auc_num_shuffles=auc_num_shuffles,
        auc_balance_classes=auc_balance_classes,
        auc_seed=auc_seed,
    )
    scm_metrics_handler = SCMMetrics(metrics=["inil"])

    # Determine output directory (used for optional cached inference)
    output_dir = resolve_output_dir(cfg, output_dir)
    cache_dir = cfg.inference.get("cache_dir", None)
    inference_root = resolve_inference_root(cfg, output_dir)
    use_cached_inference = bool(cfg.inference.get("use_cached_inference", True))
    cache_inference = bool(cfg.inference.get("cache_inference", True))
    cache_compress, cache_dtype, cache_n_samples_cfg = cache_settings(cfg)
    cache_n_samples = cache_n_samples_cfg
    if cache_n_samples is not None and cache_n_samples < n_samples:
        if rank == 0:
            log.warning(
                "cache_n_samples (%d) is smaller than inference.n_samples (%d); "
                "overriding cache_n_samples to %d for idempotent cache hits.",
                cache_n_samples,
                n_samples,
                n_samples,
            )
        cache_n_samples = n_samples
    suffix = cache_suffix(compress=cache_compress)

    model_name = get_model_name(cfg, model_unwrapped_raw)

    for name, loader in test_loaders.items():
        if rank == 0:
            log.info(f"Evaluating on {name}...")

        # Retrieve family for on-the-fly interventional evaluation
        test_families = getattr(data_module, "test_families", {}) or {}
        family = test_families.get(name)
        profile = infer_explicit_profile(name, family)
        profile_applied = apply_explicit_profile(model_unwrapped, profile)
        if rank == 0 and profile_applied and profile is not None:
            log.info(f"Applying explicit profile '{profile}' for dataset '{name}'.")

        # Reset internal state for this dataset
        metrics_handler.reset()
        scm_metrics_handler.reset()
        sampling_mode_ = sampling_mode(model_unwrapped)
        cached_context_logged = False
        sampling_context_logged = False

        with torch.no_grad():
            # For explicit/non-amortized models, prefer cached inference artifacts and avoid DataLoader+DistributedSampler padding.
            if not model_unwrapped.needs_pretraining:
                dataset = cast(MetaFixedDataset, loader.dataset)  # type: ignore[assignment]
                for idx in shard_indices(
                    len(dataset), rank=rank, world_size=dist_ctx.world_size
                ):
                    item = dataset[idx]
                    seed = int(item["seed"])
                    input_data_raw, adjacency_matrix_true = (
                        item["data"],
                        item["adjacency"],
                    )
                    input_data_norm = normalize_scm_data(input_data_raw)

                    # Find cached artifact
                    artifact_path = (
                        find_inference_artifact(
                            inference_root,
                            dataset_name=name,
                            model_name=model_name,
                            seed=seed,
                            prefer_compress=cache_compress,
                            use_model_subdir=bool(cache_dir),
                        )
                        if use_cached_inference
                        else None
                    )

                    input_data = input_data_norm.to(device).unsqueeze(0)
                    adjacency_matrix = adjacency_matrix_true.to(device).unsqueeze(0)

                    cache_hit_usable = False
                    cache_out_path = artifact_path
                    samples_for_metrics = torch.empty(0, device=device)

                    if artifact_path is not None:
                        try:
                            artifact = torch_load(artifact_path)
                        except Exception:
                            if rank == 0:
                                log.warning(
                                    "Failed to load cached artifact %s; resampling.",
                                    artifact_path,
                                    exc_info=True,
                                )
                            artifact = None

                        if artifact is not None:
                            prepared_cached, cached_n_samples = (
                                _prepare_cached_samples_for_metrics(
                                    artifact,
                                    requested_n_samples=n_samples,
                                )
                            )
                            if prepared_cached is not None:
                                if rank == 0 and not cached_context_logged:
                                    log.info(
                                        "Evaluation context: using cached inference for "
                                        "dataset=%s, device=%s, mode=%s",
                                        name,
                                        device,
                                        sampling_mode_,
                                    )
                                    cached_context_logged = True
                                samples_for_metrics = prepared_cached.to(device)
                                cache_hit_usable = True
                            else:
                                if rank == 0:
                                    log.warning(
                                        "Ignoring incompatible cached artifact %s "
                                        "(cached_n_samples=%s, requested_n_samples=%d); "
                                        "resampling and refreshing cache.",
                                        artifact_path,
                                        cached_n_samples,
                                        n_samples,
                                    )

                    if not cache_hit_usable:
                        if rank == 0 and not sampling_context_logged:
                            log.info(
                                "Evaluation sampling context: dataset=%s, device=%s, "
                                "n_samples=%d, mode=%s",
                                name,
                                device,
                                n_samples,
                                sampling_mode_,
                            )
                            sampling_context_logged = True
                        samples = model_unwrapped.sample(
                            input_data, num_samples=n_samples
                        )
                        samples_for_metrics = samples.permute(1, 0, 2, 3)

                        if cache_inference:
                            if cache_out_path is None:
                                cache_out_path = (
                                    (inference_root / model_name / name)
                                    if cache_dir
                                    else (inference_root / name)
                                ) / f"seed_{seed}{suffix}"

                            cached_samples = prepare_graph_samples_for_cache(
                                samples.detach(),
                                dtype=cache_dtype,
                                max_samples=cache_n_samples,
                            )
                            atomic_torch_save(
                                {
                                    "seed": seed,
                                    "idx": int(idx),
                                    "graph_samples": cached_samples.cpu(),
                                    "true_adj": (adjacency_matrix_true.detach() > 0.5)
                                    .to(dtype=torch.uint8)
                                    .cpu(),
                                    "cache_dtype": str(cache_dtype),
                                    "cache_n_samples": (
                                        int(cache_n_samples)
                                        if cache_n_samples is not None
                                        else None
                                    ),
                                    "requested_n_samples": int(n_samples),
                                    "num_samples_stored": int(cached_samples.shape[1]),
                                },
                                cache_out_path,
                            )

                    # Update handlers
                    metrics_handler.update(adjacency_matrix, samples_for_metrics)
                    if bool(getattr(model_unwrapped, "estimates_scm", False)):
                        scm_metrics_handler.update(
                            obs_data=input_data.squeeze(0),
                            graph_samples=samples_for_metrics.squeeze(1),
                            family=family,
                            seeds=[seed],
                        )

            else:
                for batch_idx, batch in enumerate(loader):
                    input_data, node_mask = _prepare_amortized_model_input(
                        batch, device
                    )
                    adjacency_matrix = batch["adjacency"].to(device)
                    seeds = batch.get("seed")
                    if seeds is not None and hasattr(seeds, "tolist"):
                        seeds = seeds.tolist()

                    samples = model_unwrapped.sample(
                        input_data,
                        num_samples=n_samples,
                        mask=node_mask,
                    )  # (Batch, n_samples, N, N)
                    samples_for_metrics = samples.permute(
                        1, 0, 2, 3
                    )  # (n_samples, Batch, N, N)

                    metrics_handler.update(adjacency_matrix, samples_for_metrics)

                    # Batch update for SCM metrics (assuming evaluation loop is usually batch size 1)
                    if bool(getattr(model_unwrapped, "estimates_scm", False)):
                        batch_seeds = seeds if isinstance(seeds, list) else None
                        if batch_seeds is None:
                            log.warning(
                                "Skipping SCM metrics because batch seeds are unavailable."
                            )
                        else:
                            for b in range(int(input_data.shape[0])):
                                scm_metrics_handler.update(
                                    obs_data=input_data[b],
                                    graph_samples=samples_for_metrics[:, b],
                                    family=family,
                                    seeds=[int(batch_seeds[b])],
                                )

        # Compute Summary Stats (Mean + SEM) and Gather Raw Data
        summary = metrics_handler.compute(summary_stats=True)
        final_metrics = metrics_handler.gather_raw_results()

        scm_summary = scm_metrics_handler.compute(summary_stats=True)
        scm_raw = scm_metrics_handler.get_raw_results()

        if rank == 0:
            # Merge
            summary.update(scm_summary)
            for k, v in scm_raw.items():
                final_metrics[k] = v

            all_summary_metrics[name] = summary
            all_raw_metrics[name] = final_metrics

            log.info(f"Results for {name}: {summary}")

            if logger:
                # Log means to Logger
                logger.log_metrics({f"test/{name}/{k}": v for k, v in summary.items()})

    # Save
    if rank == 0:
        dm_config = getattr(data_module, "config", None)
        batch_size_test = (
            int(getattr(dm_config, "batch_size_test", 1))
            if dm_config is not None
            else 1
        )
        batch_size_test_interventional = (
            int(getattr(dm_config, "batch_size_test_interventional", 1))
            if dm_config is not None
            else 1
        )

        # Graph-metric raw values are appended per update call. For amortized
        # models this equals per-batch unless batch_size_test=1.
        raw_granularity = "per_task"
        if model_unwrapped.needs_pretraining and batch_size_test > 1:
            raw_granularity = "per_batch"

        metadata = {
            "run_id": output_dir.name,
            "run_name": str(cfg.get("name", "")),
            "model_name": str(model_name),
            "output_dir": str(output_dir),
            "inference_root": str(inference_root.expanduser().resolve()),
            "inference_layout": "model_dataset" if cache_dir else "dataset",
            "inference_n_samples": int(n_samples),
            "cache_n_samples": (
                int(cache_n_samples) if cache_n_samples is not None else None
            ),
            "batch_size_test": batch_size_test,
            "batch_size_test_interventional": batch_size_test_interventional,
            "raw_granularity": raw_granularity,
        }
        model_results_path = output_dir / "metrics.json"
        with open(model_results_path, "w") as f:
            json.dump(
                {
                    "metadata": metadata,
                    "family_metadata": family_metadata,
                    "distances": family_distances,
                    "summary": all_summary_metrics,
                    "raw": all_raw_metrics,
                },
                f,
                cls=NpEncoder,
                indent=4,
            )

        log.info(f"Metrics saved to {model_results_path}")
