import torch
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from causal_meta.runners.metrics.graph import Metrics, expected_shd, expected_f1_score, log_prob_graph_scores

def test_expected_shd_correctness():
    # Batch=1, Nodes=2
    # Target: 0 -> 1
    target = torch.tensor([[[0, 1], [0, 0]]]).float()
    
    # Samples: 
    # 1. 0 -> 1 (Correct) -> SHD = 0
    # 2. 1 -> 0 (Reversed) -> SHD = 2 (1 FN, 1 FP)
    # 3. No edges -> SHD = 1 (1 FN)
    samples = torch.tensor([
        [[[0, 1], [0, 0]]],
        [[[0, 0], [1, 0]]],
        [[[0, 0], [0, 0]]]
    ]).float() # (S=3, B=1, N=2, N=2)
    
    val = expected_shd(target, samples)
    # Mean: (0 + 2 + 1) / 3 = 1.0
    assert torch.allclose(val, torch.tensor([1.0]))

def test_expected_f1_correctness():
    target = torch.tensor([[[0, 1], [0, 0]]]).float()
    samples = torch.tensor([
        [[[0, 1], [0, 0]]], # TP=1, FP=0, FN=0 -> F1=1.0
        [[[1, 1], [0, 0]]], # TP=1, FP=1, FN=0 -> F1=2/3
        [[[0, 0], [0, 0]]]  # TP=0, FP=0, FN=1 -> F1=0.0
    ]).float()
    
    val = expected_f1_score(target, samples)
    # Mean: (1.0 + 0.666... + 0.0) / 3 = 0.555...
    assert torch.allclose(val, torch.tensor([5/9]))

def test_metrics_handler_compute():
    metrics = Metrics(metrics=["e-shd", "e-edgef1"])
    target = torch.tensor([[[0, 1], [0, 0]]]).float()
    samples = torch.tensor([
        [[[0, 1], [0, 0]]],
        [[[0, 0], [0, 0]]]
    ]).float()
    
    results = metrics.compute(None, target, samples=samples)
    assert "e-shd" in results
    assert "e-edgef1" in results
    # e-shd: (0 + 1)/2 = 0.5
    # e-f1: (1 + 0)/2 = 0.5
    assert results["e-shd"] == 0.5
    assert results["e-edgef1"] == 0.5

@patch("torch.distributed.is_available")
@patch("torch.distributed.is_initialized")
@patch("torch.distributed.all_reduce")
def test_metrics_sync_distributed(mock_all_reduce, mock_init, mock_avail):
    mock_avail.return_value = True
    mock_init.return_value = True
    
    metrics_handler = Metrics()
    local_metrics = {"e-shd": 1.0, "graph_nll": 5.0}
    
    # Mock all_reduce to simulate averaging across 2 ranks
    # In reality all_reduce modifies in-place.
    def side_effect(tensor, op=None):
        tensor.div_(2) # Simulate ReduceOp.AVG (sum then div) if we pretend sum was done
        return None
    
    # Actually, the real Metrics.sync calls all_reduce with AVG.
    # So we just verify it's called.
    
    synced = metrics_handler.sync(local_metrics)
    
    assert mock_all_reduce.called
    assert "e-shd" in synced
    assert "graph_nll" in synced

@patch("torch.distributed.is_available")
@patch("torch.distributed.is_initialized")
@patch("torch.distributed.get_world_size")
@patch("torch.distributed.all_gather_object")
def test_metrics_gather_distributed(mock_all_gather, mock_world_size, mock_init, mock_avail):
    mock_avail.return_value = True
    mock_init.return_value = True
    mock_world_size.return_value = 2
    
    # Simulate two ranks gathering their lists
    rank0_data = {"e-shd": [0.5, 1.0]}
    rank1_data = {"e-shd": [1.5, 2.0]}
    
    def side_effect(gathered_list, local_obj):
        gathered_list[0] = rank0_data
        gathered_list[1] = rank1_data
        return None
    
    mock_all_gather.side_effect = side_effect
    
    metrics_handler = Metrics()
    gathered = metrics_handler.gather(rank0_data) # Rank 0 calls it
    
    assert len(gathered["e-shd"]) == 4
    assert gathered["e-shd"] == [0.5, 1.0, 1.5, 2.0]
