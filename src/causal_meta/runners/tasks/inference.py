from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.distributed as dist
from omegaconf import DictConfig

from causal_meta.datasets.data_module import CausalMetaModule
from causal_meta.datasets.torch_datasets import MetaFixedDataset
from causal_meta.models.base import BaseModel
from causal_meta.runners.logger.base import BaseLogger
from causal_meta.runners.tasks.utils import (
    infer_device,
    resolve_inference_root,
    sampling_mode,
    shard_indices,
)
from causal_meta.runners.utils.artifacts import (
    atomic_torch_save,
    cache_settings,
    cache_suffix,
    get_model_name,
    prepare_graph_samples_for_cache,
    resolve_output_dir,
)
from causal_meta.runners.utils.distributed import DistributedContext
from causal_meta.runners.utils.explicit_profiles import (
    apply_explicit_profile,
    infer_explicit_profile,
)

log = logging.getLogger(__name__)


def _all_reduce_sum(value: int, device: torch.device) -> int:
    """Sum a scalar value across all ranks."""
    if not (dist.is_available() and dist.is_initialized()):
        return value
    tensor = torch.tensor(value, device=device, dtype=torch.long)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return int(tensor.item())


def _batch_indices(indices: range, batch_size: int) -> List[List[int]]:
    """Split indices into batches of at most batch_size."""
    idx_list = list(indices)
    return [idx_list[i : i + batch_size] for i in range(0, len(idx_list), batch_size)]


def run(
    cfg: DictConfig,
    model: BaseModel,
    data_module: CausalMetaModule,
    *,
    logger: BaseLogger | None = None,
    output_dir: Path | None = None,
) -> Dict[str, int]:
    """
    Execute inference for non-amortized/explicit models and cache graph samples to disk.

    This task is designed to decouple expensive posterior sampling (e.g., MCMC/VI)
    from metric evaluation. It iterates through the test datasets, runs the model
    to generate graph samples, and saves the results as artifacts.

    Supports batched inference via ``inference.batch_size`` config (default: 1).
    Batching reduces Python overhead and enables parallelization in model internals.

    Args:
        cfg: Experiment configuration.
        model: The model instance (usually needs_pretraining=False).
        data_module: The data module providing test datasets.
        logger: Optional logger (unused in this task but kept for signature consistency).
        output_dir: Directory to save inference artifacts.

    Returns:
        Dict[str, int]: A mapping of dataset names to the number of samples generated.
    """
    dist_ctx = DistributedContext.current()
    rank = dist_ctx.rank
    world_size = dist_ctx.world_size

    if model.needs_pretraining:
        if rank == 0:
            log.info("Skipping inference: model.needs_pretraining=True.")
        return {}

    # Ensure datasets are initialized
    if not getattr(data_module, "test_datasets", None):
        data_module.setup()

    # Determine inference root directory
    cache_dir = cfg.get("inference", {}).get("cache_dir", None)
    inference_root = resolve_inference_root(cfg, resolve_output_dir(cfg, output_dir))

    model_name = get_model_name(cfg, model)

    inference_root.mkdir(parents=True, exist_ok=True)

    # Best-effort device inference (explicit models might have no parameters)
    device = infer_device(model, dist_ctx)

    n_samples = int(cfg.get("inference", {}).get("n_samples", 100))
    inference_batch_size = int(cfg.get("inference", {}).get("batch_size", 1))
    if inference_batch_size < 1:
        inference_batch_size = 1

    cache_compress, cache_dtype, cache_n_samples = cache_settings(cfg)
    suffix = cache_suffix(compress=cache_compress)

    written: Dict[str, int] = {}

    test_datasets: Dict[str, MetaFixedDataset] = getattr(data_module, "test_datasets")
    test_families = getattr(data_module, "test_families", {}) or {}
    for name, dataset in test_datasets.items():
        profile = infer_explicit_profile(name, test_families.get(name))
        profile_applied = apply_explicit_profile(model, profile)
        if rank == 0:
            log.info(f"Running inference cache for dataset '{name}'...")
            if profile_applied and profile is not None:
                log.info(f"Applying explicit profile '{profile}' for dataset '{name}'.")
            if inference_batch_size > 1:
                log.info(f"Using inference batch_size={inference_batch_size}")

        # If using a shared/persistent cache_dir, keep a model namespace.
        # For per-run artifacts, keep files directly under the model's run folder.
        out_dir = (
            (inference_root / model_name / name)
            if cache_dir
            else (inference_root / name)
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        total = len(dataset)
        local_count = 0
        log_interval = max(1, total // (world_size * 10))  # Log ~10 times per rank

        # Get sharded indices and split into batches
        sharded_indices = shard_indices(len(dataset), rank=rank, world_size=world_size)
        batches = _batch_indices(sharded_indices, inference_batch_size)
        sampling_mode_ = sampling_mode(model)
        sampling_context_logged = False

        for batch_indices in batches:
            # Collect batch items, filtering out already-cached entries
            batch_items: List[Tuple[int, int, torch.Tensor, torch.Tensor, Path]] = []
            for idx in batch_indices:
                item = dataset[idx]
                seed = int(item["seed"])
                out_path = out_dir / f"seed_{seed}{suffix}"
                if out_path.exists():
                    local_count += 1
                    continue
                input_data, adjacency_matrix = item["data"], item["adjacency"]
                batch_items.append((idx, seed, input_data, adjacency_matrix, out_path))

            if not batch_items:
                continue

            # Stack inputs for batched inference
            inputs = torch.stack([item[2] for item in batch_items], dim=0).to(device)

            if rank == 0 and not sampling_context_logged:
                log.info(
                    "Inference sampling context: dataset=%s, device=%s, "
                    "batch_size=%d, n_samples=%d, mode=%s",
                    name,
                    device,
                    int(inputs.shape[0]),
                    n_samples,
                    sampling_mode_,
                )
                sampling_context_logged = True

            with torch.no_grad():
                # samples: (batch_size, n_samples, num_nodes, num_nodes)
                samples = model.sample(inputs, num_samples=n_samples)

            # Unbatch and save each result
            for i, (idx, seed, _, adjacency_matrix, out_path) in enumerate(batch_items):
                sample_i = samples[
                    i : i + 1
                ]  # Keep batch dim for prepare_graph_samples
                artifact = {
                    "seed": seed,
                    "idx": int(idx),
                    "graph_samples": prepare_graph_samples_for_cache(
                        sample_i.detach(),
                        dtype=cache_dtype,
                        max_samples=cache_n_samples,
                    ).cpu(),
                    "true_adj": (adjacency_matrix.detach() > 0.5)
                    .to(dtype=torch.uint8)
                    .cpu(),
                    "cache_dtype": str(cache_dtype),
                    "cache_n_samples": (
                        int(cache_n_samples) if cache_n_samples is not None else None
                    ),
                }

                atomic_torch_save(artifact, out_path)
                local_count += 1

                # Periodic progress logging (rank-local only to avoid collective
                # deadlocks — all_reduce requires all ranks to call it together,
                # but local_count differs per rank).
                if rank == 0 and logger and local_count % log_interval == 0:
                    logger.log_metrics(
                        {
                            f"inference/{name}/local_completed": local_count,
                            f"inference/{name}/total": total,
                        }
                    )

        # Final aggregation across all ranks
        global_count = _all_reduce_sum(local_count, device)
        written[name] = global_count

        if logger and rank == 0:
            logger.log_metrics(
                {
                    f"inference/{name}/completed": global_count,
                    f"inference/{name}/total": total,
                    f"inference/{name}/progress": global_count / max(1, total),
                }
            )

    if rank == 0:
        log.info(f"Inference artifacts written under {inference_root}")

    return written
