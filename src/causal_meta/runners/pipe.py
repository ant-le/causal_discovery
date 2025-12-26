import os
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import wandb
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from causal_meta.runners.tasks import pre_training, evaluation, inference
from causal_meta.datasets.data_module import CausalMetaModule
from causal_meta.models.factory import ModelFactory

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../../../experiments", config_name="examples/full_experiment")
def main(cfg: DictConfig):
    # 0. DDP Setup
    is_distributed = False
    local_rank = 0
    
    if "LOCAL_RANK" in os.environ:
        is_distributed = True
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        
    # Configure logging to be less verbose on non-zero ranks
    if is_distributed and dist.get_rank() != 0:
        logging.getLogger().setLevel(logging.WARNING)

    if not is_distributed or dist.get_rank() == 0:
        log.info("Starting Pipeline...")
    
    # 1. Logger Setup (Only on Rank 0)
    use_wandb = cfg.get("logger", {}).get("wandb", {}).get("enabled", False)
    if use_wandb and (not is_distributed or dist.get_rank() == 0):
        log.info("Initializing WandB...")
        wandb_cfg = cfg.logger.wandb
        wandb.init(
            project=wandb_cfg.get("project", "causal_meta"),
            entity=wandb_cfg.get("entity", None),
            name=wandb_cfg.get("name", cfg.name),
            config=OmegaConf.to_container(cfg, resolve=True),
            tags=wandb_cfg.get("tags", []),
            mode=wandb_cfg.get("mode", "online"),
        )
    
    # 2. Setup Data
    if not is_distributed or dist.get_rank() == 0:
        log.info("Initializing Data Module...")
    
    data_module = CausalMetaModule.from_config(cfg.data)
    # Note: data_module setup might need to be called on all ranks if it sets up shared state,
    # but currently CausalMetaModule setup is lazy or deterministic per config.
    # However, seeds must be consistent. Config loading ensures this.

    # Resolve models
    if "models" in cfg:
        # Convert DictConfig to dict if needed, though OmegaConf handles iteration
        model_configs = cfg.models
    else:
        # Backward compatibility / Single model mode
        model_configs = {"default_model": cfg.model}

    for model_name, model_cfg in model_configs.items():
        if not is_distributed or dist.get_rank() == 0:
            log.info(f"=== Processing Model: {model_name} ===")
        
        # Instantiate Model
        model_cfg_container = OmegaConf.to_container(model_cfg, resolve=True)
        model = ModelFactory.create(model_cfg_container)
        
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        model.to(device)

        needs_pretraining = model.needs_pretraining
        
        if is_distributed:
            # Wrap model in DDP
            # find_unused_parameters might be needed if some branches (like decoder) aren't used in every step
            model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

        # 3/4. Pre-training OR Inference
        if needs_pretraining:
            pre_training.run(cfg, model, data_module) # train+val dataset
        else:
            inference.run(cfg, model, data_module) # test dataset (once per test set)

        if not is_distributed or dist.get_rank() == 0:
            inference_type = "Pre-Training" if needs_pretraining else "Inference"
            log.info(f"--- Phase 1: {inference_type} ({model_name}) ---")
        
        # 5. Evaluation
        if not is_distributed or dist.get_rank() == 0:
            log.info(f"--- Phase 2: Evaluation ({model_name}) ---")
        evaluation.run(cfg, model, data_module) # test dataset
    
    if not is_distributed or dist.get_rank() == 0:
        log.info("Pipeline Finished.")
    
    if use_wandb and (not is_distributed or dist.get_rank() == 0):
        wandb.finish()
        
    if is_distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
