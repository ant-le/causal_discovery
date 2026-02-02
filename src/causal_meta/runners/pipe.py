import logging
import os
from pathlib import Path

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from torch.nn.parallel import DistributedDataParallel as DDP

from causal_meta.datasets.data_module import CausalMetaModule
from causal_meta.models.factory import ModelFactory
from causal_meta.runners.logger.base import BaseLogger
from causal_meta.runners.logger.local import LocalLogger
from causal_meta.runners.logger.wandb import WandbLogger
from causal_meta.runners.tasks import evaluation, inference, pre_training
from causal_meta.runners.utils.distributed import DistributedContext
from causal_meta.runners.utils.env import log_environment_info
from causal_meta.runners.utils.seeding import (get_experiment_seed,
                                               seed_everything)

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg: DictConfig):
    run_pipeline(cfg)


def run_pipeline(cfg: DictConfig):
    def _validate_experiment_config(cfg_obj: DictConfig) -> None:
        missing = []
        for key in ["name", "data", "trainer", "inference"]:
            if not hasattr(cfg_obj, key):
                missing.append(key)
        if not hasattr(cfg_obj, "model"):
            missing.append("model")
        if missing:
            raise ValueError(f"Missing required config keys: {missing}")

        trainer = cfg_obj.trainer
        for key in ["lr", "max_steps"]:
            if not hasattr(trainer, key):
                raise ValueError(f"Missing required trainer key: trainer.{key}")

        model_cfg = cfg_obj.model
        if not hasattr(model_cfg, "type"):
            raise ValueError("Missing required model key: model.type")

    # 0. Validate config early (fail fast on cluster)
    _validate_experiment_config(cfg)

    # 1. Distributed setup
    dist_ctx = DistributedContext.setup()
    is_distributed = dist_ctx.is_distributed
    local_rank = dist_ctx.local_rank
    use_wandb = cfg.get("logger", {}).get("wandb", {}).get("enabled", False)
    logger: BaseLogger | None = None

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
        tf32_enabled = bool(cfg.trainer.get("tf32", False))
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = tf32_enabled
            torch.backends.cudnn.allow_tf32 = tf32_enabled

        # 3. Setup Data
        if (not is_distributed) or dist_ctx.is_main_process:
            log.info("Initializing Data Module...")

        data_module = CausalMetaModule.from_config(cfg.data)

        override_dir = cfg.get("inference", {}).get("output_dir", None)
        if override_dir:
            base_output_dir = Path(str(override_dir))
        else:
            try:
                base_output_dir = Path(HydraConfig.get().runtime.output_dir)
            except Exception:
                base_output_dir = Path(os.getcwd())

        base_output_dir.mkdir(parents=True, exist_ok=True)

        model_specific_cfg = cfg.model
        model_id = str(getattr(model_specific_cfg, "type", "model"))
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
                if k not in {"trainer", "inference"}
            }
            model = ModelFactory.create(model_params)

            device = dist_ctx.device
            model.to(device)

            if is_distributed:
                model = DDP(
                    model,
                    device_ids=[local_rank] if torch.cuda.is_available() else None,
                    output_device=local_rank if torch.cuda.is_available() else None,
                    find_unused_parameters=False,
                )

            # 5/6. Pre-training OR Inference
            model_unwrapped = model.module if is_distributed else model
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
