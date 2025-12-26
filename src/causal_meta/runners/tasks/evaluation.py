import json
import logging
import os
from collections import defaultdict
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

def gather_results(local_results: dict) -> dict:
    """Gather lists of metrics from all ranks and concatenate them."""
    if not (dist.is_available() and dist.is_initialized()):
        return local_results

    # Gather all results objects (list of dicts)
    world_size = dist.get_world_size()
    gathered_results = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_results, local_results)

    # Merge: {metric: [val, ...]}
    merged = defaultdict(list)
    for res in gathered_results:
        if res:
            for k, v in res.items():
                merged[k].extend(v)
    
    return dict(merged)

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
    
    all_summary_metrics = {}
    all_raw_metrics = {}

    n_samples = cfg.inference.get("n_samples", 100)
    
    for name, loader in test_loaders.items():
        if rank == 0:
            log.info(f"Evaluating on {name}...")
        
        # Local storage for this rank
        local_metrics = defaultdict(list)
        
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
                
                # Metrics (single task per batch assumed, so we take index 0 or mean)
                # compute_graph_metrics returns dict of scalars if batch=1
                m = compute_graph_metrics(probs, adj)
                
                # Probabilistic metrics
                m["e_shd"] = float(np.mean(expected_shd(adj, samples_for_metrics).cpu().numpy()))
                m["e_f1"] = float(
                    np.mean(expected_f1_score(adj, samples_for_metrics).cpu().numpy())
                )
                m["graph_log_prob"] = float(np.mean(log_prob_graph_scores(adj, samples_for_metrics)))
                m["graph_auroc"] = float(np.mean(auc_graph_scores(adj, samples_for_metrics)))
                
                for k, v in m.items():
                    local_metrics[k].append(v)
        
        # Sync across ranks (gather full lists)
        final_metrics = gather_results(dict(local_metrics))
        
        if rank == 0:
            # Compute Summary Stats (Mean + SEM)
            summary = {}
            for k, v in final_metrics.items():
                arr = np.array(v)
                summary[f"{k}_mean"] = float(np.mean(arr))
                summary[f"{k}_sem"] = float(np.std(arr, ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
                summary[f"{k}_std"] = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
            
            all_summary_metrics[name] = summary
            all_raw_metrics[name] = final_metrics
            
            log.info(f"Results for {name}: {summary}")
            
            if wandb.run:
                # Log means to WandB
                wandb.log({f"test/{name}/{k}": v for k, v in summary.items()})

    # Save
    if rank == 0:
        output_dir = Path(os.getcwd()) / "results"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save Summary
        with open(output_dir / "metrics_summary.json", "w") as f:
            json.dump(all_summary_metrics, f, cls=NpEncoder, indent=4)
            
        # Save Raw (for Box Plots)
        with open(output_dir / "metrics_raw.json", "w") as f:
            json.dump(all_raw_metrics, f, cls=NpEncoder, indent=4)
        
        log.info(f"Metrics saved to {output_dir}")
