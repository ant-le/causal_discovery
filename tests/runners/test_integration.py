import torch
import pytest
from causal_meta.runners.metrics.eval import compute_graph_metrics
from causal_meta.models.avici.model import AviciModel

def test_avici_loss():
    # Setup
    model = AviciModel(num_nodes=5, d_model=16, nhead=2, num_layers=2)
    
    # Batch=2, N=5
    logits = torch.randn(2, 5, 5)
    target = torch.randint(0, 2, (2, 5, 5)).float()
    
    loss = model.calculate_loss(logits, target)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0

def test_eval_metrics():
    # Perfect prediction
    probs = torch.tensor([[[0.1, 0.9], [0.1, 0.1]]]) # (1, 2, 2)
    target = torch.tensor([[[0, 1], [0, 0]]])
    
    metrics = compute_graph_metrics(probs, target, threshold=0.5)
    assert metrics["shd"] == 0
    assert metrics["f1"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0

    # Bad prediction
    probs = torch.tensor([[[0.9, 0.1], [0.9, 0.9]]])
    target = torch.tensor([[[0, 1], [0, 0]]])
    
    metrics = compute_graph_metrics(probs, target, threshold=0.5)
    # Predicted: [[1, 0], [1, 1]] (3 edges)
    # True: [[0, 1], [0, 0]] (1 edge)
    # TP: 0
    # FP: 3 (indices (0,0), (1,0), (1,1))
    # FN: 1 (index (0,1))
    # SHD: 4
    assert metrics["shd"] == 4
    assert metrics["f1"] == 0.0

def test_training_step_integration():
    # Setup
    model = AviciModel(num_nodes=5, d_model=16, nhead=2, num_layers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Fake batch
    batch_size = 2
    n_samples = 10
    n_nodes = 5
    x = torch.randn(batch_size, n_samples, n_nodes)
    adj = torch.randint(0, 2, (batch_size, n_nodes, n_nodes)).float()
    
    # Forward
    output = model(x)
    assert output.shape == (batch_size, n_nodes, n_nodes)
    
    # Loss
    loss = model.calculate_loss(output, adj)
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    assert not torch.isnan(loss)

