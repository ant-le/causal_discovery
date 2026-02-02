from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict

import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from causal_meta.datasets.data_module import CausalMetaModule
from causal_meta.datasets.torch_datasets import MetaFixedDataset
from causal_meta.models.base import BaseModel
from causal_meta.runners.utils.artifacts import (
    atomic_torch_save, cache_settings, cache_suffix,
    prepare_graph_samples_for_cache)
from causal_meta.runners.utils.distributed import DistributedContext

log = logging.getLogger(__name__)


def _get_output_dir(cfg: DictConfig) -> Path:
    override = cfg.get("inference", {}).get("output_dir", None)
    if override:
        return Path(str(override))
    try:
        return Path(HydraConfig.get().runtime.output_dir)
    except Exception:
        return Path(os.getcwd())


def _shard_indices(n: int, rank: int, world_size: int) -> range:
    # No padding, deterministic sharding: rank i handles i, i+world_size, ...
    return range(rank, n, world_size)


def run(
    cfg: DictConfig,
    model: BaseModel,
    data_module: CausalMetaModule,
    *,
    logger=None,
    output_dir: Path | None = None,
) -> Dict[str, int]:
    """
    Execute inference for non-amortized/explicit models and cache graph samples to disk.

    This task is designed to decouple expensive posterior sampling (e.g., MCMC/VI)
    from metric evaluation. It iterates through the test datasets, runs the model
    to generate graph samples, and saves the results as artifacts.

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

    if output_dir is None:
        output_dir = _get_output_dir(cfg)
    else:
        output_dir = Path(output_dir)
    inference_root = output_dir / "inference"
    inference_root.mkdir(parents=True, exist_ok=True)

    # Best-effort device inference (explicit models might have no parameters)
    params = list(model.parameters())
    device = (
        params[0].device
        if params
        else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
    )

    n_samples = int(cfg.get("inference", {}).get("n_samples", 100))
    cache_compress, cache_dtype, cache_n_samples = cache_settings(cfg)
    suffix = cache_suffix(compress=cache_compress)

    written: Dict[str, int] = {}

    test_datasets: Dict[str, MetaFixedDataset] = getattr(data_module, "test_datasets")
    for name, dataset in test_datasets.items():
        if rank == 0:
            log.info(f"Running inference cache for dataset '{name}'...")

        out_dir = inference_root / name
        out_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        for idx in _shard_indices(len(dataset), rank=rank, world_size=world_size):
            item = dataset[idx]
            seed = int(item["seed"])
            input_data, adjacency_matrix = item["data"], item["adjacency"]
            input_data = input_data.to(device)
            model_input = input_data.unsqueeze(0)

            with torch.no_grad():
                samples = model.sample(
                    model_input, num_samples=n_samples
                )  # (1, K, N, N)

            artifact = {
                "seed": seed,
                "idx": int(idx),
                "graph_samples": prepare_graph_samples_for_cache(
                    samples.detach(), dtype=cache_dtype, max_samples=cache_n_samples
                ).cpu(),
                "true_adj": (adjacency_matrix.detach() > 0.5)
                .to(dtype=torch.uint8)
                .cpu(),
                "cache_dtype": str(cache_dtype),
                "cache_n_samples": (
                    int(cache_n_samples) if cache_n_samples is not None else None
                ),
            }

            atomic_torch_save(artifact, out_dir / f"seed_{seed}{suffix}")
            count += 1

        written[name] = count

    if rank == 0:
        log.info(f"Inference artifacts written under {inference_root}")

    return written
