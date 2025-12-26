import logging
import os

import wandb

import torch
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from omegaconf import DictConfig
from pathlib import Path

from causal_meta.models.base import BaseModel
from causal_meta.datasets.data_module import CausalMetaModule
from causal_meta.runners.metrics.eval import compute_graph_metrics

log = logging.getLogger(__name__)

def run(cfg: DictConfig, model: BaseModel, data_module: CausalMetaModule):
    # 0. DDP Utilities
    is_distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if is_distributed else 0
    
    # 1. Setup
    if rank == 0:
        log.info(f"Starting experiment: {cfg.name}")
        log.info(f"Working directory: {os.getcwd()}")
    
    # Device is already set in pipe.py, but we can double check or get it from model
    device = next(model.parameters()).device
    
    # 2. Data
    # train_dataloader handles DDP seeding internally
    train_loader = data_module.train_dataloader()
    
    # Unwrap model for attribute access
    model_unwrapped = model.module if is_distributed else model
    
    if not model_unwrapped.needs_pretraining:
        if rank == 0:
            log.info("Model does not require pre-training. Exiting.")
        return

    # 3. Optimizer
    optimizer = optim.Adam(model.parameters(), lr=cfg.trainer.lr)
    
    # 4. Training Loop
    max_steps = cfg.trainer.max_steps
    log_every_n_steps = cfg.trainer.get("log_every_n_steps", 100)
    val_check_interval = cfg.trainer.get("val_check_interval", 1000)
    regulariser_update_interval = cfg.trainer.get("regulariser_update_interval", 0)
    
    best_val_metric = float("-inf")
    step = 0
    train_iter = iter(train_loader)
    
    if rank == 0:
        log.info("Starting training loop...")
    
    path = Path("checkpoints")
    if rank == 0:
        path.mkdir(parents=True, exist_ok=True)

    while step < max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
            
        x, adj = batch
        x = x.to(device)
        adj = adj.to(device)
        
        # Forward
        model.train()
        optimizer.zero_grad()
        output = model(x)
        
        # Loss
        next_step = step + 1
        update_regulariser = (
            regulariser_update_interval > 0 and next_step % regulariser_update_interval == 0
        )
        
        # Call calculate_loss on the unwrapped model
        # The output tensor from DDP forward is on the correct device
        loss = model_unwrapped.calculate_loss(output, adj, update_regulariser=update_regulariser)

        loss.backward()
        optimizer.step()
        
        step += 1
        
        # Logging
        if step % log_every_n_steps == 0:
            if rank == 0:
                log_str = f"Step {step}/{max_steps} | Loss: {loss.item():.4f}"
                log.info(log_str)
                
                if wandb.run:
                    wandb.log({"train/loss": loss.item(), "train/step": step})
            
        # Validation
        if step % val_check_interval == 0:
            val_n_samples = int(cfg.get("inference", {}).get("n_samples", 10))
            # Validate involves sampling, which might need unwrapped model or handling in validate
            val_metrics = validate(model_unwrapped, data_module, device, n_samples=val_n_samples)
            
            # Aggregate metrics if DDP (validate runs on local chunk)
            # For simplicity, we can just log rank 0 or average them. 
            # Since validation set is split, average is approximation if sizes differ slightly.
            # But DistributedSampler handles splitting. We should gather metrics.
            # For now, let's just log what we have on rank 0 or implement rudimentary sync?
            # Actually, proper validation in DDP requires gathering.
            # Given complexity, we will rely on WandB to aggregate or rank 0 to be representative enough?
            # Or we implement sync. Let's assume sync for scalar metrics.
            
            if is_distributed:
                val_metrics = sync_metrics(val_metrics)

            if rank == 0:
                log.info(f"Validation at step {step}: {val_metrics}")
                
                if wandb.run:
                    wandb.log({f"val/{k}": v for k, v in val_metrics.items()}, step=step)
                
                # Save Last
                save_checkpoint(model_unwrapped, optimizer, step, path / "last.pt")
                
                # Save Best (using F1 as metric)
                current_metric = val_metrics.get("mean_f1", 0.0)
                if current_metric > best_val_metric:
                    best_val_metric = current_metric
                    save_checkpoint(model_unwrapped, optimizer, step, path / "best.pt")
                    log.info(f"New best model saved! F1: {best_val_metric:.4f}")

    if rank == 0:
        log.info("Training finished.")
    
    # Reload Best Model (Only on Rank 0, then broadcast? Or just load on all?)
    # Since we saved only on rank 0, only rank 0 can load easily. 
    # But usually all ranks need the weights.
    # We should load on rank 0 and broadcast, or ensure file system is shared and load on all.
    # Assuming shared FS.
    
    if is_distributed:
        dist.barrier() # Wait for rank 0 to save
        
    best_path = path / "best.pt"
    if best_path.exists():
        if rank == 0:
            log.info(f"Reloading best model from {best_path}")
        
        # Map location to local device
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank} if is_distributed else device
        checkpoint = torch.load(best_path, map_location=map_location)
        model_unwrapped.load_state_dict(checkpoint['model_state_dict'])
    
    if is_distributed:
        dist.barrier()

def validate(model, data_module, device, n_samples: int = 10):
    model.eval()
    val_loaders = data_module.val_dataloader()
    metrics_sum = {}
    counts = {}
    
    with torch.no_grad():
        for name, loader in val_loaders.items():
            for batch in loader:
                x, adj = batch
                x = x.to(device)
                adj = adj.to(device)
                
                samples = model.sample(x, n_samples=n_samples)
                probs = samples.float().mean(dim=1)
                
                m = compute_graph_metrics(probs, adj)
                
                for k, v in m.items():
                    key = f"{name}/{k}"
                    metrics_sum[key] = metrics_sum.get(key, 0.0) + v
                    counts[key] = counts.get(key, 0) + 1
                    
                    # Global aggregate
                    metrics_sum[k] = metrics_sum.get(k, 0.0) + v
                    counts[k] = counts.get(k, 0) + 1

    avg_metrics = {k: v / counts[k] for k, v in metrics_sum.items()}
    
    # Calculate mean F1 across all datasets
    avg_metrics["mean_f1"] = avg_metrics.get("f1", 0.0)
    
    return avg_metrics

def sync_metrics(metrics: dict) -> dict:
    """Synchronize metrics across ranks by averaging."""
    keys = sorted(metrics.keys())
    values = torch.tensor([metrics[k] for k in keys], device='cuda')
    dist.all_reduce(values, op=dist.ReduceOp.AVG)
    return {k: v.item() for k, v in zip(keys, values)}

def save_checkpoint(model, optimizer, step, filepath):
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, filepath)

if __name__ == "__main__":
    main()
