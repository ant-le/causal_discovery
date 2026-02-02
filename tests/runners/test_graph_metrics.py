import numpy as np
import torch

from causal_meta.runners.metrics.graph import (
    expected_f1_score,
    expected_shd,
    log_prob_graph_scores,
    ancestor_f1_score,
)


def test_ancestor_f1() -> None:
    # 0 -> 1 -> 2
    target = torch.tensor([[[0, 1, 0], [0, 0, 1], [0, 0, 0]]]).float()
    
    # Correct structure
    pred_correct = torch.tensor([[[[0, 1, 0], [0, 0, 1], [0, 0, 0]]]]).float()
    # Missing 1->2 but adds 0->2 directly
    pred_shortcut = torch.tensor([[[[0, 0, 1], [0, 0, 0], [0, 0, 0]]]]).float()
    
    # For target, ancestors are (0,1), (1,2), (0,2)
    # Correct pred: ancestors match.
    f1_correct = ancestor_f1_score(target, pred_correct)
    assert f1_correct[0].item() == 1.0
    
    # Shortcut pred: ancestors are (0,2). 
    # TP = {(0,2)}, FP = {}, FN = {(0,1), (1,2)}
    # F1 = 2*1 / (2*1 + 0 + 2) = 2/4 = 0.5
    f1_shortcut = ancestor_f1_score(target, pred_shortcut)
    assert np.isclose(f1_shortcut[0].item(), 0.5)


def test_expected_shd_and_expected_f1() -> None:
    target = torch.tensor([[[0, 1], [0, 0]]])  # (B=1, N=2, N=2)
    pred = torch.tensor(
        [
            [[[0, 1], [0, 0]]],  # perfect
            [[[0, 0], [1, 0]]],  # reversed
        ]
    )  # (S=2, B=1, N=2, N=2)

    shd = expected_shd(target, pred)
    assert shd.shape == (1,)
    # SHD: perfect=0, reversed=1 (if not double) or 2 (if double). 
    # Current expected_shd uses abs diff which is double counting mismatches in DAG context if considering reversed edge as 2 errors (FP+FN).
    # target=[0,1], pred=[0,0] -> FN=1
    # target=[0,0], pred=[1,0] -> FP=1
    # Total diff = 2.
    # So for sample 2, SHD=2.
    # Mean SHD = (0 + 2) / 2 = 1.0.
    assert shd[0].item() == 1.0

    f1 = expected_f1_score(target, pred)
    assert f1.shape == (1,)
    # Sample 1: Perfect match -> F1=1.0
    # Sample 2: 
    # TP = 0 (target has (0,1), pred has (1,0))
    # FP = 1 (pred has (1,0))
    # FN = 1 (target has (0,1))
    # F1 = 0 / (0 + 1 + 1) = 0.0
    # Mean F1 = (1.0 + 0.0) / 2 = 0.5
    assert np.isclose(f1[0].item(), 0.5)


def test_log_prob_graph_scores() -> None:
    targets = torch.tensor([[[0.0, 1.0], [0.0, 0.0]]])
    preds = torch.stack([targets, targets], dim=0)  # (S=2, B=1, N=2, N=2)

    log_probs = log_prob_graph_scores(targets, preds)
    assert len(log_probs) == 1
    assert np.isfinite(log_probs[0])
    # Probability is 1.0 (clamped), log prob should be near 0
    assert log_probs[0] > -1e-3


