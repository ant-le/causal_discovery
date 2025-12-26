import numpy as np
import torch

from causal_meta.runners.metrics.eval import (
    auc_graph_scores,
    balance_for_auc,
    calc_SHD,
    calculate_auc,
    expected_f1_score,
    expected_shd,
    log_prob_graph_scores,
)


def test_calc_shd_reverse_edge_counts_once_when_not_double() -> None:
    target = np.array([[0, 1], [0, 0]])
    pred = np.array([[0, 0], [1, 0]])
    assert calc_SHD(target, pred, double_for_anticausal=True) == 2
    assert calc_SHD(target, pred, double_for_anticausal=False) == 1


def test_expected_shd_and_expected_f1() -> None:
    target = np.array([[[0, 1], [0, 0]]])  # (B=1, N=2, N=2)
    pred = np.array(
        [
            [[[0, 1], [0, 0]]],  # perfect
            [[[0, 0], [1, 0]]],  # reversed
        ]
    )  # (S=2, B=1, N=2, N=2)

    shd = expected_shd(target, pred)
    assert shd.shape == (1,)
    assert shd[0] == 1.0

    f1 = expected_f1_score(target, pred)
    assert f1.shape == (1,)
    assert np.isclose(f1[0], 0.5)


def test_auc_and_log_prob_graph_scores() -> None:
    targets = torch.tensor([[[0.0, 1.0], [0.0, 0.0]]])
    preds = torch.stack([targets, targets], dim=0)  # (S=2, B=1, N=2, N=2)

    aucs = auc_graph_scores(targets, preds)
    assert len(aucs) == 1
    assert np.isclose(aucs[0], 1.0)

    log_probs = log_prob_graph_scores(targets, preds)
    assert len(log_probs) == 1
    assert np.isfinite(log_probs[0])
    assert log_probs[0] > -1e-3


def test_balance_for_auc_balances_even_imbalance_and_flips_scores() -> None:
    rng = np.random.default_rng(0)
    target = np.array([-1, -1, -1, -1, 1, 1])
    pred_scores = np.array([1.0, 2.0, -3.0, 4.0, 5.0, -6.0])

    balanced_target, balanced_scores = balance_for_auc(target, pred_scores, rng=rng)
    assert int(np.sum(balanced_target)) == 0

    flipped = np.nonzero(balanced_target != target)[0]
    assert flipped.shape == (1,)
    idx = int(flipped[0])
    assert target[idx] == -1
    assert balanced_target[idx] == 1
    assert balanced_scores[idx] == -pred_scores[idx]


def test_calculate_auc_is_deterministic_and_does_not_mutate_inputs() -> None:
    target = np.array([-1, 1, -1, 1], dtype=int)
    pred_scores = np.array([-0.2, 0.3, -0.1, 0.4], dtype=float)
    target_copy = target.copy()
    pred_copy = pred_scores.copy()

    auc1 = calculate_auc(target, pred_scores, num_shuffles=10, rng=np.random.default_rng(123))
    auc2 = calculate_auc(target, pred_scores, num_shuffles=10, rng=np.random.default_rng(123))
    assert 0.0 <= auc1 <= 1.0
    assert np.isclose(auc1, auc2)

    assert np.array_equal(target, target_copy)
    assert np.array_equal(pred_scores, pred_copy)

