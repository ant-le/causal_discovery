from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from causal_meta.datasets.data_module import CausalMetaModule
from causal_meta.datasets.torch_datasets import MetaFixedDataset
from causal_meta.models.base import BaseModel
from causal_meta.runners.metrics.graph import Metrics
from causal_meta.runners.metrics.scm import SCMMetrics
from causal_meta.runners.utils.artifacts import (
    atomic_torch_save, cache_settings, cache_suffix, find_inference_artifact,
    prepare_graph_samples_for_cache, torch_load)
from causal_meta.runners.utils.distributed import DistributedContext

log = logging.getLogger(__name__)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def run(
    cfg: DictConfig,
    model: BaseModel,
    data_module: CausalMetaModule,
    *,
    logger=None,
    output_dir: Path | None = None,
):
    # DDP Setup
    dist_ctx = DistributedContext.current()
    is_distributed = dist_ctx.is_distributed
    rank = dist_ctx.rank

    if rank == 0:
        log.info(f"Starting evaluation: {cfg.name}")

    # Device (explicit models may have no parameters)
    params = list(model.parameters())
    device = params[0].device if params else dist_ctx.device

    # Data
    test_loaders = data_module.test_dataloader()

    model.eval()

    # Unwrap if DDP
    model_unwrapped = model.module if is_distributed else model

    all_summary_metrics = {}
    all_raw_metrics = {}

    n_samples = int(cfg.inference.get("n_samples", 100))

    # Initialize Metrics Handlers
    metrics_handler = Metrics(
        metrics=[
            "e-shd",
            "e-edgef1",
            "e-sid",
            "graph_nll",
            "edge_entropy",
            "ancestor_f1",
            "auc",
        ]
    )
    scm_metrics_handler = SCMMetrics(metrics=["inil"])

    # Determine output directory (used for optional cached inference)
    if output_dir is None:
        try:
            output_dir = Path(HydraConfig.get().runtime.output_dir)
        except Exception:
            output_dir = Path(os.getcwd())
    else:
        output_dir = Path(output_dir)
    inference_root = output_dir / "inference"
    use_cached_inference = bool(cfg.inference.get("use_cached_inference", True))
    cache_inference = bool(cfg.inference.get("cache_inference", True))
    cache_compress, cache_dtype, cache_n_samples = cache_settings(cfg)
    suffix = cache_suffix(compress=cache_compress)

    def _shard_indices(n: int, rank: int, world_size: int) -> range:
        return range(rank, n, world_size)

    for name, loader in test_loaders.items():
        if rank == 0:
            log.info(f"Evaluating on {name}...")

        # Retrieve family for on-the-fly interventional evaluation
        test_families = getattr(data_module, "test_families", {}) or {}
        family = test_families.get(name)

        # Reset internal state for this dataset
        metrics_handler.reset()
        scm_metrics_handler.reset()

        with torch.no_grad():
            # For explicit/non-amortized models, prefer cached inference artifacts and avoid DataLoader+DistributedSampler padding.
            if not model_unwrapped.needs_pretraining:
                dataset: MetaFixedDataset = loader.dataset  # type: ignore[assignment]
                for idx in _shard_indices(
                    len(dataset), rank=rank, world_size=dist_ctx.world_size
                ):
                    item = dataset[idx]
                    seed = int(item["seed"])
                    input_data_raw, adjacency_matrix_true = (
                        item["data"],
                        item["adjacency"],
                    )

                    # Find cached artifact
                    artifact_path = (
                        find_inference_artifact(
                            inference_root,
                            dataset_name=name,
                            seed=seed,
                            prefer_compress=cache_compress,
                        )
                        if use_cached_inference
                        else None
                    )

                    input_data = input_data_raw.to(device).unsqueeze(0)
                    adjacency_matrix = adjacency_matrix_true.to(device).unsqueeze(0)

                    if artifact_path is not None:
                        artifact = torch_load(artifact_path)
                        graph_samples = artifact["graph_samples"].to(device)
                        samples_for_metrics = graph_samples.permute(1, 0, 2, 3)
                    else:
                        samples = model_unwrapped.sample(
                            input_data, num_samples=n_samples
                        )
                        samples_for_metrics = samples.permute(1, 0, 2, 3)

                        if (
                            cache_inference
                            and find_inference_artifact(
                                inference_root,
                                dataset_name=name,
                                seed=seed,
                                prefer_compress=cache_compress,
                            )
                            is None
                        ):
                            out_path = inference_root / name / f"seed_{seed}{suffix}"
                            atomic_torch_save(
                                {
                                    "seed": seed,
                                    "idx": int(idx),
                                    "graph_samples": prepare_graph_samples_for_cache(
                                        samples.detach(),
                                        dtype=cache_dtype,
                                        max_samples=cache_n_samples,
                                    ).cpu(),
                                    "true_adj": (adjacency_matrix_true.detach() > 0.5)
                                    .to(dtype=torch.uint8)
                                    .cpu(),
                                    "cache_dtype": str(cache_dtype),
                                    "cache_n_samples": (
                                        int(cache_n_samples)
                                        if cache_n_samples is not None
                                        else None
                                    ),
                                },
                                out_path,
                            )

                    # Update handlers
                    metrics_handler.update(adjacency_matrix, samples_for_metrics)
                    if bool(getattr(model_unwrapped, "estimates_scm", False)):
                        scm_metrics_handler.update(
                            obs_data=input_data.squeeze(0),
                            graph_samples=samples_for_metrics.squeeze(1),
                            family=family,
                            seeds=[seed],
                            prefix=name,
                        )

            else:
                for batch_idx, batch in enumerate(loader):
                    input_data = batch["data"].to(device)
                    adjacency_matrix = batch["adjacency"].to(device)
                    seeds = batch.get("seed")
                    if seeds is not None and hasattr(seeds, "tolist"):
                        seeds = seeds.tolist()

                    samples = model_unwrapped.sample(
                        input_data, num_samples=n_samples
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
                                    prefix=name,
                                )

        # Compute Summary Stats (Mean + SEM) and Gather Raw Data
        summary = metrics_handler.compute(summary_stats=True)
        final_metrics = metrics_handler.get_raw_results()

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
        results_dir = output_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        # Save Summary
        with open(results_dir / "metrics_summary.json", "w") as f:
            json.dump(all_summary_metrics, f, cls=NpEncoder, indent=4)

        # Save Raw (for Box Plots)
        with open(results_dir / "metrics_raw.json", "w") as f:
            json.dump(all_raw_metrics, f, cls=NpEncoder, indent=4)

        log.info(f"Metrics saved to {results_dir}")
