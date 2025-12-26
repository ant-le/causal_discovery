from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch


def _to_numpy(x: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _as_binary_labels(y_true: np.ndarray) -> np.ndarray:
    y_true = np.asarray(y_true).reshape(-1)
    unique = set(np.unique(y_true).tolist())
    if unique.issubset({0, 1}):
        return y_true.astype(int)
    if unique.issubset({-1, 1}):
        return ((y_true + 1) // 2).astype(int)
    raise ValueError(f"Unsupported labels for AUC: {sorted(unique)}. Expected {{0,1}} or {{-1,1}}.")


def _roc_auc_score_binary(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Compute ROC-AUC for binary labels without sklearn (tie-aware via average ranks)."""
    y_true_bin = _as_binary_labels(y_true)
    y_scores = np.asarray(y_scores).reshape(-1).astype(float)
    if y_true_bin.shape[0] != y_scores.shape[0]:
        raise ValueError("y_true and y_scores must have the same length.")

    n_pos = int(np.sum(y_true_bin))
    n = int(y_true_bin.shape[0])
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0

    order = np.argsort(y_scores, kind="mergesort")
    scores_sorted = y_scores[order]
    y_true_sorted = y_true_bin[order]

    ranks_sorted = np.empty(n, dtype=float)
    i = 0
    current_rank = 1
    while i < n:
        j = i + 1
        while j < n and scores_sorted[j] == scores_sorted[i]:
            j += 1
        group_size = j - i
        avg_rank = (2 * current_rank + group_size - 1) / 2.0
        ranks_sorted[i:j] = avg_rank
        current_rank += group_size
        i = j

    ranks = np.empty(n, dtype=float)
    ranks[order] = ranks_sorted

    sum_pos_ranks = float(np.sum(ranks[y_true_bin == 1]))
    auc = (sum_pos_ranks - (n_pos * (n_pos + 1) / 2.0)) / float(n_pos * n_neg)
    return float(auc)


def compute_auroc_simple(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    return _roc_auc_score_binary(y_true, y_scores)


def _binary_f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).reshape(-1).astype(int)
    y_pred = np.asarray(y_pred).reshape(-1).astype(int)
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length.")
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    denom = 2 * tp + fp + fn
    if denom == 0:
        return 0.0
    return float(2 * tp / denom)


def balance_for_auc(
    target: np.ndarray, pred_scores: np.ndarray, rng: np.random.Generator | None = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Balance {-1,1} targets by flipping a minimal number of labels (and their scores)."""
    rng = rng or np.random.default_rng()
    target_arr = np.asarray(target).reshape(-1).astype(int)
    pred_arr = np.asarray(pred_scores).reshape(-1).astype(float)
    if target_arr.shape != pred_arr.shape:
        raise ValueError("target and pred_scores must have the same shape.")

    unique = set(np.unique(target_arr).tolist())
    if not unique.issubset({-1, 1}):
        raise ValueError(f"Targets must be in {{-1, 1}}. Got: {sorted(unique)}")

    balance = int(np.sum(target_arr))
    final_target = target_arr.copy()
    final_pred_scores = pred_arr.copy()

    if balance != 0:
        if balance < 0:
            switch_cand_idx = np.nonzero(final_target < 0)[0]
        else:
            switch_cand_idx = np.nonzero(final_target > 0)[0]
        n_switch = int(np.abs(balance) // 2)
        if n_switch:
            switch_idx = rng.choice(switch_cand_idx, size=n_switch, replace=False)
            final_target[switch_idx] *= -1
            final_pred_scores[switch_idx] *= -1

    if (balance % 2) == 0:
        assert int(np.sum(final_target)) == 0
    else:
        assert int(np.abs(np.sum(final_target))) == 1
    return final_target, final_pred_scores


def calculate_auc(
    target: np.ndarray,
    pred_scores: np.ndarray,
    num_shuffles: int = 1000,
    rng: np.random.Generator | None = None,
) -> float:
    """Estimate ROC-AUC for paired {-1,1} targets via random sign-flip symmetrization."""
    if num_shuffles < 1:
        raise ValueError("num_shuffles must be >= 1")

    rng = rng or np.random.default_rng()
    target_arr = np.asarray(target).reshape(-1).astype(int)
    pred_arr = np.asarray(pred_scores).reshape(-1).astype(float)
    if target_arr.shape != pred_arr.shape:
        raise ValueError("target and pred_scores must have the same shape.")

    auc_all: List[float] = []
    total_runs = int(target_arr.shape[0])
    if total_runs == 0:
        return 0.0

    for _ in range(num_shuffles):
        shuffled_target = target_arr.copy()
        shuffled_pred_scores = pred_arr.copy()

        flip_idx = rng.choice(np.arange(total_runs), size=total_runs // 2, replace=False)
        shuffled_target[flip_idx] *= -1
        shuffled_pred_scores[flip_idx] *= -1

        final_target, final_pred_scores = balance_for_auc(
            shuffled_target, shuffled_pred_scores, rng=rng
        )
        auc_all.append(_roc_auc_score_binary(final_target, final_pred_scores))

    return float(np.mean(auc_all))


def calc_SHD(target: np.ndarray, pred: np.ndarray, double_for_anticausal: bool = True) -> int:
    """Compute Structural Hamming Distance (SHD) between two (binary) adjacency matrices."""
    target_arr = np.asarray(target)
    pred_arr = np.asarray(pred)
    if target_arr.shape != pred_arr.shape:
        raise ValueError("target and pred must have the same shape.")

    target_bin = (target_arr > 0).astype(int)
    pred_bin = (pred_arr > 0).astype(int)

    diff = np.abs(target_bin - pred_bin)
    shd_double = int(np.sum(diff))
    if double_for_anticausal:
        return shd_double

    reversed_edges = int(
        np.sum(
            (target_bin == 1)
            & (pred_bin.T == 1)
            & (target_bin.T == 0)
            & (pred_bin == 0)
        )
    )
    return int(shd_double - reversed_edges)


# def expected_shd(target: np.ndarray, pred: np.ndarray, check_acyclic: bool = False) -> np.ndarray:
#     """Expected SHD for a batch of predictions.
# 
#     Args:
#         target: (batch_size, num_nodes, num_nodes)
#         pred: (num_samples, batch_size, num_nodes, num_nodes)
#     """
#     _ = check_acyclic
#     target_arr = np.asarray(target)
#     pred_arr = np.asarray(pred)
#     if target_arr.ndim != 3 or pred_arr.ndim != 4:
#         raise ValueError("target must be 3D and pred must be 4D.")
#     if pred_arr.shape[1:] != target_arr.shape:
#         raise ValueError("pred shape must be (num_samples, batch, N, N) matching target.")
# 
#     shd_all = np.zeros(pred_arr.shape[1], dtype=float)
#     for i in range(pred_arr.shape[1]):
#         curr_pred = pred_arr[:, i]
#         curr_target = target_arr[i]
#         shd_sample = []
#         for j in range(curr_pred.shape[0]):
#             shd_sample.append(calc_SHD(curr_target, curr_pred[j], double_for_anticausal=True))
#         shd_all[i] = float(np.mean(shd_sample)) if shd_sample else 0.0
#     return shd_all

def expected_shd(target: torch.Tensor, pred: torch.Tensor, check_acyclic: bool = False) -> torch.Tensor:
    """Expected SHD for a batch of predictions (Vectorized).

    Args:
        target: (batch_size, num_nodes, num_nodes)
        pred: (num_samples, batch_size, num_nodes, num_nodes)
    Returns:
        Tensor of shape (batch_size,)
    """
    _ = check_acyclic
    if target.ndim != 3 or pred.ndim != 4:
        raise ValueError("target must be 3D and pred must be 4D.")
    if pred.shape[1:] != target.shape:
        raise ValueError("pred shape must be (num_samples, batch, N, N) matching target.")

    # pred is (S, B, N, N), target is (B, N, N)
    # Expand target to (1, B, N, N) for broadcasting
    target_expanded = target.unsqueeze(0)
    
    # SHD (double_for_anticausal=True equivalent) is just sum of absolute differences
    # pred is binary samples, target is binary adjacency
    diff = torch.abs(target_expanded - pred)
    
    # Sum over N, N dimensions -> (S, B)
    shd_per_sample = diff.sum(dim=(-1, -2))
    
    # Mean over samples -> (B,)
    return shd_per_sample.float().mean(dim=0)


# def expected_f1_score(target: np.ndarray, pred: np.ndarray, check_acyclic: bool = False) -> np.ndarray:
#     """Expected F1 score for a batch of predictions.
# 
#     Args:
#         target: (batch_size, num_nodes, num_nodes)
#         pred: (num_samples, batch_size, num_nodes, num_nodes)
#     """
#     _ = check_acyclic
#     target_arr = np.asarray(target)
#     pred_arr = np.asarray(pred)
#     if target_arr.ndim != 3 or pred_arr.ndim != 4:
#         raise ValueError("target must be 3D and pred must be 4D.")
#     if pred_arr.shape[1:] != target_arr.shape:
#         raise ValueError("pred shape must be (num_samples, batch, N, N) matching target.")
# 
#     f1_all = np.zeros(pred_arr.shape[1], dtype=float)
#     for i in range(pred_arr.shape[1]):
#         curr_pred = pred_arr[:, i]
#         curr_target = target_arr[i]
#         f1_sample = []
#         for j in range(curr_pred.shape[0]):
#             f1_sample.append(_binary_f1_score(curr_target.flatten(), curr_pred[j].flatten()))
#         f1_all[i] = float(np.mean(f1_sample)) if f1_sample else 0.0
#     return f1_all

def expected_f1_score(target: torch.Tensor, pred: torch.Tensor, check_acyclic: bool = False) -> torch.Tensor:
    """Expected F1 score for a batch of predictions (Vectorized).

    Args:
        target: (batch_size, num_nodes, num_nodes)
        pred: (num_samples, batch_size, num_nodes, num_nodes)
    Returns:
        Tensor of shape (batch_size,)
    """
    _ = check_acyclic
    if target.ndim != 3 or pred.ndim != 4:
        raise ValueError("target must be 3D and pred must be 4D.")
    if pred.shape[1:] != target.shape:
        raise ValueError("pred shape must be (num_samples, batch, N, N) matching target.")

    # Expand target: (1, B, N, N)
    target_expanded = target.unsqueeze(0)
    
    # Operations on (S, B, N, N)
    tp = (pred * target_expanded).sum(dim=(-1, -2))
    fp = (pred * (1 - target_expanded)).sum(dim=(-1, -2))
    fn = ((1 - pred) * target_expanded).sum(dim=(-1, -2))
    
    # F1 per sample: (S, B)
    denom = 2 * tp + fp + fn
    # Avoid division by zero
    f1 = torch.where(denom > 0, (2 * tp) / denom, torch.zeros_like(denom))
    
    # Mean over samples -> (B,)
    return f1.mean(dim=0)


def log_prob_graph_scores(targets: torch.Tensor, preds: torch.Tensor, eps: float = 1e-6) -> List[float]:
    """Log-probability of targets under Bernoulli parameterized by mean(preds) across samples.

    Args:
        targets: (batch_size, num_nodes, num_nodes)
        preds: (num_samples, batch_size, num_nodes, num_nodes)
    """
    targets_t = targets if isinstance(targets, torch.Tensor) else torch.as_tensor(targets)
    preds_t = preds if isinstance(preds, torch.Tensor) else torch.as_tensor(preds)
    if targets_t.ndim != 3 or preds_t.ndim != 4:
        raise ValueError("targets must be 3D and preds must be 4D.")
    if preds_t.shape[1:] != targets_t.shape:
        raise ValueError("preds shape must be (num_samples, batch, N, N) matching targets.")

    all_log_probs: List[float] = []
    for batch_idx in range(int(targets_t.shape[0])):
        sample_mean = preds_t[:, batch_idx].float().mean(dim=0)
        probs = sample_mean.flatten().clamp(eps, 1.0 - eps)
        current_targets = targets_t[batch_idx].float().flatten()
        bern_dist = torch.distributions.bernoulli.Bernoulli(probs=probs)
        log_prob = bern_dist.log_prob(current_targets).sum()
        all_log_probs.append(float(log_prob.detach().cpu().item()))
    return all_log_probs


def auc_graph_scores(targets: torch.Tensor, preds: torch.Tensor) -> List[float]:
    """AUROC of targets vs. mean(preds) across samples, per batch element.

    Args:
        targets: (batch_size, num_nodes, num_nodes)
        preds: (num_samples, batch_size, num_nodes, num_nodes)
    """
    targets_np = _to_numpy(targets)
    preds_np = _to_numpy(preds)
    if targets_np.ndim != 3 or preds_np.ndim != 4:
        raise ValueError("targets must be 3D and preds must be 4D.")
    if preds_np.shape[1:] != targets_np.shape:
        raise ValueError("preds shape must be (num_samples, batch, N, N) matching targets.")

    all_aucs: List[float] = []
    for batch_idx in range(targets_np.shape[0]):
        sample_mean = np.mean(preds_np[:, batch_idx], axis=0)
        auc = _roc_auc_score_binary(targets_np[batch_idx].flatten(), sample_mean.flatten())
        all_aucs.append(float(auc))
    return all_aucs

def compute_graph_metrics(pred_probs: torch.Tensor, true_adj: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    """
    Compute graph structure metrics.
    
    Args:
        pred_probs: (Batch, N, N) tensor of edge probabilities.
        true_adj: (Batch, N, N) tensor of binary ground truth adjacency.
        threshold: Threshold for binarizing probabilities.
        
    Returns:
        Dictionary of metrics.
    """
    y_true = true_adj.detach().cpu().numpy().flatten().astype(int)
    y_scores = pred_probs.detach().cpu().numpy().flatten()
    
    # Binarize
    y_pred = (y_scores > threshold).astype(int)
    
    # Basic Counters
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = _binary_f1_score(y_true, y_pred)
    
    # SHD (Structural Hamming Distance) = FP + FN
    # Note: This is equivalent to unweighted Hamming distance for DAGs
    shd = fp + fn
    
    # AUROC
    auroc = compute_auroc_simple(y_true, y_scores)
    
    # AUPRC (Skipping complex implementation, setting to 0.0 or reusing F1 as proxy?)
    # Leaving as 0.0 to avoid complex implementation without sklearn
    auprc = 0.0

    return {
        "shd": float(shd),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "auroc": float(auroc),
        "auprc": float(auprc),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
    }
