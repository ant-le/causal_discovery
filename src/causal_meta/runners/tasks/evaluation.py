import json
import logging
import os
from omegaconf import DictConfig
import torch
import torch.distributed as dist
from pathlib import Path
import numpy as np
import wandb

from causal_meta.datasets.data_module import CausalMetaModule
from causal_meta.models.base import BaseModel
from causal_meta.runners.metrics.eval import (
    auc_graph_scores,
    compute_graph_metrics,
    expected_f1_score,
    expected_shd,
    log_prob_graph_scores,
)

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

def sync_metrics(metrics: dict) -> dict:
    """Synchronize metrics across ranks by averaging."""
    keys = sorted(metrics.keys())
    values = torch.tensor([metrics[k] for k in keys], device='cuda')
    dist.all_reduce(values, op=dist.ReduceOp.AVG)
    return {k: v.item() for k, v in zip(keys, values)}

def run(cfg: DictConfig, model: BaseModel, data_module: CausalMetaModule):
    # DDP Setup
    is_distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if is_distributed else 0
    
    if rank == 0:
        log.info(f"Starting evaluation: {cfg.name}")
    
    # Device
    device = next(model.parameters()).device
    
    # Data
    test_loaders = data_module.test_dataloader()
    
    model.eval()
    
    # Unwrap if DDP
    model_unwrapped = model.module if is_distributed else model
    
    all_metrics = {}

    n_samples = cfg.inference.get("n_samples", 100)
    
    for name, loader in test_loaders.items():
        if rank == 0:
            log.info(f"Evaluating on {name}...")
        metrics_sum = {}
        counts = {}
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                x, adj = batch
                x = x.to(device)
                adj = adj.to(device)
                
                # Use unwrapped model for sampling logic which might be custom
                samples = model_unwrapped.sample(x, n_samples=n_samples)  # (Batch, n_samples, N, N)
                samples_for_metrics = samples.permute(1, 0, 2, 3)  # (n_samples, Batch, N, N)

                # Use posterior mean edge probabilities (estimated from samples)
                probs = samples.float().mean(dim=1)
                
                # Metrics
                m = compute_graph_metrics(probs, adj)
                m["e_shd"] = float(np.mean(expected_shd(adj.cpu().numpy(), samples_for_metrics.cpu().numpy())))
                m["e_f1"] = float(
                    np.mean(expected_f1_score(adj.cpu().numpy(), samples_for_metrics.cpu().numpy()))
                )
                m["graph_log_prob"] = float(np.mean(log_prob_graph_scores(adj, samples_for_metrics)))
                m["graph_auroc"] = float(np.mean(auc_graph_scores(adj, samples_for_metrics)))
                
                for k, v in m.items():
                    metrics_sum[k] = metrics_sum.get(k, 0.0) + v
                    counts[k] = counts.get(k, 0) + 1
        
        # Local Average
        dataset_metrics = {k: v / counts[k] for k, v in metrics_sum.items()}
        
        # Sync if distributed
        if is_distributed:
            dataset_metrics = sync_metrics(dataset_metrics)
            
        all_metrics[name] = dataset_metrics
        
        if rank == 0:
            log.info(f"Results for {name}: {dataset_metrics}")
            if wandb.run:
                wandb.log({f"test/{name}/{k}": v for k, v in dataset_metrics.items()})

    # Save
    if rank == 0:
        output_dir = Path(os.getcwd()) / "results"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / "metrics.json", "w") as f:
            json.dump(all_metrics, f, cls=NpEncoder, indent=4)
        
        log.info(f"Metrics saved to {output_dir / 'metrics.json'}")
