from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Mapping, cast

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, open_dict
from torch.nn.parallel import DistributedDataParallel as DDP

from causal_meta.datasets.data_module import CausalMetaModule
from causal_meta.datasets.generators.configs import (ErdosRenyiConfig,
                                                     MixtureGraphConfig,
                                                     SBMConfig,
                                                     ScaleFreeConfig)
from causal_meta.datasets.generators.factory import load_data_module_config
from causal_meta.models.base import BaseModel
from causal_meta.models.factory import ModelFactory
from causal_meta.runners.logger.base import BaseLogger
from causal_meta.runners.logger.local import LocalLogger
from causal_meta.runners.logger.wandb import WandbLogger
from causal_meta.runners.tasks import analysis, evaluation, inference, pre_training
from causal_meta.runners.utils.artifacts import resolve_output_dir
from causal_meta.runners.utils.distributed import DistributedContext
from causal_meta.runners.utils.env import log_environment_info
from causal_meta.runners.utils.seeding import (get_experiment_seed,
                                               seed_everything)

log = logging.getLogger(__name__)


def _attach_file_logger(output_dir: Path, *, filename: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / filename
    root_logger = logging.getLogger()

    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler):
            try:
                if Path(handler.baseFilename) == log_path:
                    return
            except Exception:
                continue

    formatter = logging.Formatter(
        "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"
    )
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)


def _estimate_sbm_edge_probability(cfg: SBMConfig, n_nodes: int) -> float:
    """Estimate expected directed edge probability for the SBM generator."""
    n_blocks = int(cfg.n_blocks)
    if n_blocks < 1:
        return 0.0

    base_size, remainder = divmod(int(n_nodes), n_blocks)
    block_sizes = [base_size + (1 if i < remainder else 0) for i in range(n_blocks)]

    expected_edges = 0.0
    for i in range(n_blocks):
        size_i = float(block_sizes[i])
        expected_edges += (size_i * (size_i - 1.0) / 2.0) * float(cfg.p_intra)
        for j in range(i + 1, n_blocks):
            size_j = float(block_sizes[j])
            expected_edges += size_i * size_j * float(cfg.p_inter)

    max_edges = float(n_nodes * (n_nodes - 1) / 2)
    if max_edges <= 0.0:
        return 0.0
    return float(max(0.0, min(1.0, expected_edges / max_edges)))


def _estimate_edge_probability_from_graph_cfg(graph_cfg: object, n_nodes: int) -> float:
    """Estimate expected edge probability from a graph config object."""
    max_edges = float(n_nodes * (n_nodes - 1) / 2)
    if max_edges <= 0.0:
        return 0.0

    if isinstance(graph_cfg, ErdosRenyiConfig):
        p = graph_cfg.edge_prob
        if p is None:
            p = graph_cfg.sparsity
        if p is None:
            raise ValueError("ErdosRenyiConfig must define edge_prob or sparsity.")
        return float(max(0.0, min(1.0, float(p))))

    if isinstance(graph_cfg, ScaleFreeConfig):
        m = int(graph_cfg.m)
        expected_edges = float(m * n_nodes - (m * (m + 1) / 2.0))
        return float(max(0.0, min(1.0, expected_edges / max_edges)))

    if isinstance(graph_cfg, SBMConfig):
        return _estimate_sbm_edge_probability(graph_cfg, n_nodes)

    if isinstance(graph_cfg, MixtureGraphConfig):
        if len(graph_cfg.generators) != len(graph_cfg.weights):
            raise ValueError("MixtureGraphConfig generators/weights length mismatch.")
        total = float(sum(graph_cfg.weights))
        if total <= 0.0:
            raise ValueError("MixtureGraphConfig requires positive total weight.")
        probs = [
            _estimate_edge_probability_from_graph_cfg(sub_cfg, n_nodes)
            for sub_cfg in graph_cfg.generators
        ]
        weighted = sum(float(w) * float(p) for w, p in zip(graph_cfg.weights, probs))
        return float(max(0.0, min(1.0, weighted / total)))

    if not hasattr(graph_cfg, "instantiate"):
        raise ValueError(
            f"Unsupported graph config for random baseline: {type(graph_cfg)}"
        )

    generator = graph_cfg.instantiate()
    probe = 128
    base_seed = 0
    rng = np.random.default_rng(base_seed)
    torch_gen = torch.Generator().manual_seed(base_seed)
    total_edges = 0.0
    for i in range(probe):
        adjacency = generator(
            n_nodes,
            seed=base_seed + i,
            torch_generator=torch_gen,
            rng=rng,
        )
        total_edges += float(adjacency.sum().item())
    return float(max(0.0, min(1.0, total_edges / (probe * max_edges))))


def infer_random_baseline_edge_probability(cfg: DictConfig) -> float:
    """Infer a training-sparsity-matched edge probability for the random baseline.

    Args:
        cfg: Full Hydra config.

    Returns:
        Estimated expected directed edge probability of the training graph family.
    """
    data_cfg = load_data_module_config(cfg.data)
    train_family = data_cfg.train_family
    p_edge = _estimate_edge_probability_from_graph_cfg(
        train_family.graph_cfg,
        n_nodes=int(train_family.n_nodes),
    )
    if not (0.0 <= p_edge <= 1.0) or math.isnan(p_edge):
        raise ValueError(f"Invalid inferred random baseline p_edge={p_edge}.")
    return float(p_edge)


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg: DictConfig) -> None:
    run_pipeline(cfg)


def run_pipeline(cfg: DictConfig) -> None:
    # Check for analysis task (e.g. python main.py +task=analysis analysis.target_dir=...)
    if cfg.get("task") == "analysis":
        base_output_dir = resolve_output_dir(cfg)
        analysis.run(cfg, base_output_dir)
        return

    def _validate_experiment_config(cfg_obj: DictConfig) -> None:
        missing = []
        for key in ["name", "data", "inference"]:
            if not hasattr(cfg_obj, key):
                missing.append(key)
        if not hasattr(cfg_obj, "model"):
            missing.append("model")
        if missing:
            raise ValueError(f"Missing required config keys: {missing}")

        model_cfg = cfg_obj.model
        if not hasattr(model_cfg, "type"):
            raise ValueError("Missing required model key: model.type")

        # Only require optimizer/training settings for amortized models.
        # Explicit inference models (e.g., DiBS/BayesDAG) do not use trainer.*.
        amortized_model_types = {"avici", "bcnp"}
        model_type = str(getattr(model_cfg, "type", "")).lower()
        if model_type in amortized_model_types:
            if not hasattr(cfg_obj, "trainer"):
                raise ValueError(
                    "Missing required config key for amortized models: trainer"
                )
            trainer = cfg_obj.trainer
            for key in ["lr", "max_steps"]:
                if not hasattr(trainer, key):
                    raise ValueError(f"Missing required trainer key: trainer.{key}")

    def _maybe_fill_model_num_nodes(cfg_obj: DictConfig) -> None:
        model_cfg = cfg_obj.model
        if hasattr(model_cfg, "num_nodes"):
            return

        inferred = None
        data_cfg = getattr(cfg_obj, "data", None)
        if data_cfg is not None:
            # Prefer explicit shared key when present.
            for k in ["n_nodes", "num_nodes"]:
                if hasattr(data_cfg, k):
                    inferred = int(getattr(data_cfg, k))
                    break
            if inferred is None and hasattr(data_cfg, "train_family"):
                train_family = getattr(data_cfg, "train_family")
                if hasattr(train_family, "n_nodes"):
                    inferred = int(getattr(train_family, "n_nodes"))

        if inferred is None:
            raise ValueError(
                "model.num_nodes is required when it cannot be inferred from data.*.n_nodes"
            )

        with open_dict(cfg_obj):
            cfg_obj.model.num_nodes = inferred

    def _expand_shared_architecture(cfg_obj: DictConfig) -> None:
        """Allow shared amortized architecture under `model.arch.*`."""

        model_cfg = cfg_obj.model
        model_type = str(getattr(model_cfg, "type", "")).lower()
        if model_type not in {"avici", "bcnp"}:
            return
        if not hasattr(model_cfg, "arch"):
            return

        arch_cfg = getattr(model_cfg, "arch")
        if arch_cfg is None:
            return

        with open_dict(cfg_obj):
            for k, v in arch_cfg.items():
                if not hasattr(cfg_obj.model, k):
                    setattr(cfg_obj.model, k, v)
            # Avoid passing nested dicts to model constructors.
            try:
                del cfg_obj.model["arch"]
            except Exception:
                pass

    def _maybe_fill_random_edge_prior(cfg_obj: DictConfig) -> None:
        model_cfg = cfg_obj.model
        model_type = str(getattr(model_cfg, "type", "")).lower()
        if model_type != "random":
            return
        if hasattr(model_cfg, "p_edge") and getattr(model_cfg, "p_edge") is not None:
            return

        inferred_p_edge = infer_random_baseline_edge_probability(cfg_obj)
        with open_dict(cfg_obj):
            cfg_obj.model.p_edge = float(inferred_p_edge)

    # 0. Validate config early (fail fast on cluster)
    _validate_experiment_config(cfg)
    _maybe_fill_model_num_nodes(cfg)
    _expand_shared_architecture(cfg)
    _maybe_fill_random_edge_prior(cfg)

    # 1. Distributed setup
    dist_ctx = DistributedContext.setup()
    is_distributed = dist_ctx.is_distributed
    local_rank = dist_ctx.local_rank
    use_wandb = cfg.get("logger", {}).get("wandb", {}).get("enabled", False)
    logger: BaseLogger | None = None

    # Use Hydra's per-job output directory as the single artifact root.
    base_output_dir = resolve_output_dir(cfg)
    base_output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Configure logging to be less verbose on non-zero ranks
        if is_distributed and dist_ctx.rank != 0:
            logging.getLogger().setLevel(logging.WARNING)

        if (not is_distributed) or dist_ctx.is_main_process:
            log.info("Starting Pipeline...")
            log_environment_info()

        # 2. Seed process RNGs (dataset workers handle their own seeding internally)
        seed = get_experiment_seed(cfg, fallback=0)
        seed_everything(seed, deterministic=bool(cfg.get("deterministic", False)))

        # 2b. Optional TF32 (perf knob for transformer-heavy workloads)
        tf32_enabled = bool(cfg.get("trainer", {}).get("tf32", False))
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = tf32_enabled
            torch.backends.cudnn.allow_tf32 = tf32_enabled

        # 3. Setup Data
        if (not is_distributed) or dist_ctx.is_main_process:
            log.info("Initializing Data Module...")

        data_module = CausalMetaModule.from_config(cfg.data)

        model_specific_cfg = cfg.model
        model_id = str(
            getattr(
                model_specific_cfg, "id", getattr(model_specific_cfg, "type", "model")
            )
        )
        if (not is_distributed) or dist_ctx.is_main_process:
            _attach_file_logger(base_output_dir, filename="main.log")
        if (not is_distributed) or dist_ctx.is_main_process:
            log.info(f"=== Running Model: {model_id} ===")

        dist_ctx.barrier()

        # Initialize Logger once per run (Hydra multirun launches separate jobs per model).
        if (not is_distributed) or dist_ctx.is_main_process:
            if use_wandb:
                log.info("Initializing WandB Logger...")
                logger = WandbLogger(cfg, output_dir=str(base_output_dir))
            else:
                log.info("Initializing Local Logger...")
                logger = LocalLogger()
        else:
            logger = LocalLogger()

        try:
            # Instantiate Model
            model_params = {
                k: v
                for k, v in model_specific_cfg.items()
                if k not in {"trainer", "inference", "arch"}
            }

            # Keep BayesDAG side outputs inside the run folder by default.
            model_type_for_defaults = str(model_params.get("type", "")).lower()
            if model_type_for_defaults == "bayesdag" and "save_dir" not in model_params:
                model_params["save_dir"] = str(base_output_dir / "bayesdag_output")

            # For explicit inference models, allow moving model hyperparameters into
            # the inference block. Preferred: `inference.<model_type>.*`.
            # Backward-compatible: `inference.explicit.<model_type>.*`.
            model_type = str(model_params.get("type", "")).lower()
            inference_cfg = cfg.get("inference", {})
            override = None
            if isinstance(inference_cfg, Mapping):
                direct = inference_cfg.get(model_type)
                if isinstance(direct, Mapping):
                    override = direct
                else:
                    explicit = inference_cfg.get("explicit", {})
                    if isinstance(explicit, Mapping):
                        nested = explicit.get(model_type)
                        if isinstance(nested, Mapping):
                            override = nested
            if isinstance(override, Mapping):
                model_params.update(dict(override))

            model = ModelFactory.create(model_params)

            device = dist_ctx.device
            model.to(device)

            if is_distributed:
                use_cuda_ddp = dist_ctx.device.type == "cuda"
                model = DDP(
                    model,
                    device_ids=[local_rank] if use_cuda_ddp else None,
                    output_device=local_rank if use_cuda_ddp else None,
                    find_unused_parameters=False,
                )

            # 5/6. Pre-training OR Inference
            model_unwrapped_raw = model.module if is_distributed else model
            model_unwrapped = cast(BaseModel, model_unwrapped_raw)
            if not isinstance(model_unwrapped, BaseModel):
                raise TypeError(
                    "Expected model to be a BaseModel or DDP-wrapped BaseModel."
                )

            if model_unwrapped.needs_pretraining:
                pre_training.run(
                    cfg,
                    model,
                    data_module,
                    logger=logger,
                    output_dir=base_output_dir,
                )
                inference_type = "Pre-Training"
            else:
                inference.run(
                    cfg,
                    model_unwrapped,
                    data_module,
                    logger=logger,
                    output_dir=base_output_dir,
                )
                inference_type = "Inference"

            if (not is_distributed) or dist_ctx.is_main_process:
                log.info(f"--- Phase 1: {inference_type} ---")

            # 7. Evaluation
            if (not is_distributed) or dist_ctx.is_main_process:
                log.info("--- Phase 2: Evaluation ---")
            evaluation.run(
                cfg,
                model,
                data_module,
                logger=logger,
                output_dir=base_output_dir,
            )
        except Exception:
            if (not is_distributed) or dist_ctx.is_main_process:
                log.exception("Pipeline failed.")
            raise
        finally:
            if logger is not None:
                logger.finish()

        if (not is_distributed) or dist_ctx.is_main_process:
            log.info("Pipeline Finished.")
    finally:
        # Cleanup distributed context at the very end
        try:
            DistributedContext.cleanup()
        except Exception:
            pass


if __name__ == "__main__":
    main()
