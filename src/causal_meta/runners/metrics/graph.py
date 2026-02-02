from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Union

import torch

from causal_meta.datasets.scm import SCMFamily
from .base import BaseMetrics


def auc_graph_scores(targets: torch.Tensor, preds: torch.Tensor) -> torch.Tensor:
    """
    Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) for graph edges.
    
    Since we don't have sklearn, this computes AUC manually using the trapezoidal rule
    on sorted predictions.
    
    Args:
        targets: (Batch, N, N) binary adjacency matrices.
        preds: (NumSamples, Batch, N, N) binary graph samples or probabilities.
               If binary samples, we average them to get probabilities.
    
    Returns:
        Tensor of shape (Batch,) containing AUC score per graph.
    """
    if targets.ndim != 3 or preds.ndim != 4:
        raise ValueError("targets must be 3D and preds must be 4D.")

    # 1. Get predicted probabilities: Mean over samples -> (Batch, N, N)
    probs = preds.float().mean(dim=0)
    
    # 2. Flatten N,N dimensions -> (Batch, N*N)
    y_true = targets.reshape(targets.shape[0], -1).float()
    y_score = probs.reshape(probs.shape[0], -1)

    # 3. Compute AUC per batch item
    auc_scores = []
    device = targets.device
    
    for i in range(y_true.shape[0]):
        yt = y_true[i]
        ys = y_score[i]
        
        # Sort by score descending
        desc_score_indices = torch.argsort(ys, descending=True)
        yt_sorted = yt[desc_score_indices]
        ys_sorted = ys[desc_score_indices]
        
        # Total positives and negatives
        tp_total = yt_sorted.sum()
        fp_total = yt_sorted.numel() - tp_total
        
        if tp_total == 0 or fp_total == 0:
            # Undefined AUC if only one class is present.
            # Convention: return 0.5 (random guessing) or NaN.
            # Here we return 0.5.
            auc_scores.append(0.5)
            continue

        # Cumulative sums
        tps = torch.cumsum(yt_sorted, dim=0)
        fps = torch.cumsum(1 - yt_sorted, dim=0)
        
        # TPR and FPR
        tpr = tps / tp_total
        fpr = fps / fp_total
        
        # Trapezoidal rule for AUC: sum( (FPR_i - FPR_{i-1}) * TPR_i )
        # Prepend 0 to FPR and TPR for integration
        fpr_diff = torch.cat([fpr[0:1], fpr[1:] - fpr[:-1]])
        auc = torch.sum(fpr_diff * tpr).item()
        auc_scores.append(auc)

    return torch.tensor(auc_scores, device=device)


def expected_shd(
    target: torch.Tensor, pred: torch.Tensor, check_acyclic: bool = False
) -> torch.Tensor:
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
        raise ValueError(
            "pred shape must be (num_samples, batch, N, N) matching target."
        )

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


def expected_f1_score(
    target: torch.Tensor, pred: torch.Tensor, check_acyclic: bool = False
) -> torch.Tensor:
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
        raise ValueError(
            "pred shape must be (num_samples, batch, N, N) matching target."
        )

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


def graph_nll_score(
    targets: torch.Tensor, preds: torch.Tensor, eps: float = 1e-6
) -> float:
    """Negative Log-Probability of targets under Bernoulli parameterized by mean(preds).

    Args:
        targets: (batch_size, num_nodes, num_nodes)
        preds: (num_samples, batch_size, num_nodes, num_nodes)
    """
    targets_t = (
        targets if isinstance(targets, torch.Tensor) else torch.as_tensor(targets)
    )
    preds_t = preds if isinstance(preds, torch.Tensor) else torch.as_tensor(preds)

    # Mean probability per edge: (B, N, N)
    p = preds_t.float().mean(dim=0).clamp(eps, 1.0 - eps)
    y = targets_t.float()

    # Bernoulli log-likelihood per batch item (vectorized):
    # sum_{i} [ y_i * log(p_i) + (1-y_i) * log(1-p_i) ]
    log_prob = (y * torch.log(p) + (1.0 - y) * torch.log(1.0 - p)).sum(dim=(-1, -2))

    # Return mean NLL across batch
    return -float(log_prob.mean().item())


def edge_entropy(preds: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Mean entropy of edge existence probabilities across the batch.

    Args:
        preds: (num_samples, batch_size, num_nodes, num_nodes)
    Returns:
        Scalar tensor of mean entropy per potential edge.
    """
    # Mean probability per edge: (B, N, N)
    p = preds.float().mean(dim=0).clamp(eps, 1.0 - eps)

    # Binary Entropy: -p log p - (1-p) log (1-p)
    entropy = -(p * torch.log(p) + (1 - p) * torch.log(1 - p))

    # Return mean over all potential edges (B, N, N)
    return entropy.mean()


def structural_interventional_distance(
    target: torch.Tensor, pred: torch.Tensor
) -> torch.Tensor:
    """
    Structural Interventional Distance (SID), averaged over posterior samples.

    Torch-based batched implementation following the SID criterion. The inner
    computation is implemented to run efficiently on GPU via batched tensor
    operations (with a small loop over intervention nodes).

    Args:
        target: (batch_size, num_nodes, num_nodes)
        pred: (num_samples, batch_size, num_nodes, num_nodes)
    """
    if target.ndim != 3 or pred.ndim != 4:
        raise ValueError("target must be 3D and pred must be 4D.")
    if pred.shape[1:] != target.shape:
        raise ValueError(
            "pred shape must be (num_samples, batch, N, N) matching target."
        )

    num_samples, batch_size, n_nodes, _ = pred.shape

    # Flatten (S, B) pairs so we can do a single batched SID computation.
    targets_flat = target.unsqueeze(0).expand(num_samples, batch_size, n_nodes, n_nodes)
    targets_flat = targets_flat.reshape(num_samples * batch_size, n_nodes, n_nodes)
    preds_flat = pred.reshape(num_samples * batch_size, n_nodes, n_nodes)

    sid_flat = compute_sid_batched(targets_flat, preds_flat)  # (S*B,)
    sid_sb = sid_flat.view(num_samples, batch_size)
    return sid_sb.mean(dim=0)


def compute_sid_batched(adj_true: torch.Tensor, adj_est: torch.Tensor) -> torch.Tensor:
    """
    Compute Structural Interventional Distance (SID) for a batch of graph pairs.

    This follows the SID criterion in a form that is amenable to batched torch ops
    (GPU-friendly). The only Python loops are over intervention nodes (P) and a
    small fixed-point iteration bounded by P.

    Args:
        adj_true: (B, P, P) binary adjacency for true graphs G (edge i -> j is 1 at [i, j]).
        adj_est:  (B, P, P) binary adjacency for estimated graphs H.
    Returns:
        (B,) tensor with SID counts in [0, P*(P-1)].
    """
    if adj_true.ndim != 3 or adj_est.ndim != 3:
        raise ValueError("adj_true and adj_est must be 3D (B, P, P).")
    if adj_true.shape != adj_est.shape or adj_true.shape[-1] != adj_true.shape[-2]:
        raise ValueError(
            "adj_true and adj_est must have the same square shape (B, P, P)."
        )

    bsz, p, _ = adj_true.shape
    if p <= 1:
        return torch.zeros(bsz, device=adj_true.device, dtype=torch.float32)

    device = adj_true.device

    # Ensure float adjacency for matmul, but treat as boolean masks.
    g = (adj_true > 0).to(dtype=torch.float32)
    h = (adj_est > 0).to(dtype=torch.float32)

    # 1) Precompute transitive closure / reachability for G.
    # T_G[b, i, j] is True iff i is an ancestor of j in G (path length >= 1).
    t_g = _transitive_closure_bool(g)  # (B, P, P) bool
    t_g_f = t_g.to(dtype=torch.float32)

    # Ancestors in G (including self): Anc[b, i, j] True iff j is ancestor of i or j==i.
    eye = torch.eye(p, device=device, dtype=torch.bool).unsqueeze(0).expand(bsz, p, p)
    anc = t_g.transpose(1, 2) | eye

    sid_counts = torch.zeros(bsz, device=device, dtype=torch.float32)

    for i in range(p):
        # Z = Parents of i in H (column i of H).
        z_mask = h[:, :, i] > 0  # (B, P) bool

        # Check 0: reverse edges (a parent of i in H is a descendant of i in G).
        err_reverse = z_mask & t_g[:, i, :]  # (B, P) bool

        # Check 1: Condition (*) part 1 (directed paths through "bad mediators").
        w_is_anc_z = (
            torch.matmul(t_g_f, z_mask.unsqueeze(-1).to(torch.float32)).squeeze(-1)
        ) > 0
        w_is_desc_i = t_g[:, i, :]  # descendants of i in G
        bad_w = w_is_desc_i & w_is_anc_z  # nodes on a directed path from i that reach Z
        err_cond1 = (
            torch.matmul(bad_w.unsqueeze(1).to(torch.float32), t_g_f).squeeze(1)
        ) > 0

        # Check 2: Condition (*) part 2 (non-directed paths; d-connection in G_{underline{i}}).
        g_back = g.clone()
        g_back[:, i, :] = 0.0  # remove edges leaving i

        # Nodes that are in Anc(Z) (including Z itself) for each batch item.
        z_anc_mask = (
            torch.matmul(
                anc.to(torch.float32), z_mask.unsqueeze(-1).to(torch.float32)
            ).squeeze(-1)
        ) > 0

        # Batched Bayes-ball reachability (fixed-point iteration).
        curr_up = torch.zeros(bsz, p, device=device, dtype=torch.bool)
        curr_down = torch.zeros(bsz, p, device=device, dtype=torch.bool)
        curr_up[:, i] = True

        for _ in range(p):
            not_z = ~z_mask

            can_pass_down = (curr_down | curr_up) & not_z
            next_down = (
                torch.matmul(
                    can_pass_down.to(torch.float32).unsqueeze(1), g_back
                ).squeeze(1)
            ) > 0

            can_pass_up_chain = curr_up & not_z
            can_pass_up_collider = curr_down & z_anc_mask
            can_pass_up = can_pass_up_chain | can_pass_up_collider
            next_up = (
                torch.matmul(
                    can_pass_up.to(torch.float32).unsqueeze(1), g_back.transpose(1, 2)
                ).squeeze(1)
            ) > 0

            new_down = curr_down | next_down
            new_up = curr_up | next_up
            if torch.equal(new_down, curr_down) and torch.equal(new_up, curr_up):
                break
            curr_down = new_down
            curr_up = new_up

        d_connected = curr_down | curr_up
        d_connected[:, i] = False

        # Combine errors for node i.
        in_z = z_mask
        errors_in_z = in_z & err_reverse
        errors_out_z = (~in_z) & (err_cond1 | d_connected)

        total_errors = errors_in_z | errors_out_z
        total_errors[:, i] = False
        sid_counts += total_errors.sum(dim=1).to(torch.float32)

    return sid_counts


def _transitive_closure_bool(adj: torch.Tensor) -> torch.Tensor:
    """
    Compute reachability (transitive closure) for a batch of adjacency matrices.

    Args:
        adj: (..., N, N) adjacency (0/1 or bool)
    Returns:
        (..., N, N) bool reachability matrix (excluding diagonal)
    """
    if adj.ndim < 2:
        raise ValueError("adj must have at least 2 dims.")
    if adj.shape[-1] != adj.shape[-2]:
        raise ValueError("adj must be square in the last two dims.")

    n = int(adj.shape[-1])
    reach = adj.to(dtype=torch.bool)

    if n <= 1:
        return reach

    # Repeated squaring for transitive closure:
    # reach <- reach OR (reach @ reach), repeated log2(n) times is sufficient.
    steps = int(math.ceil(math.log2(n)))
    reach_f = reach.to(dtype=torch.float32)
    for _ in range(steps):
        reach = reach | (reach_f @ reach_f > 0)
        reach_f = reach.to(dtype=torch.float32)

    eye = torch.eye(n, device=adj.device, dtype=torch.bool)
    return reach & ~eye


def ancestor_f1_score(target: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """F1 score for causal ancestor relationships.

    Checks if i is an ancestor of j in both graphs.
    """
    if target.ndim != 3 or pred.ndim != 4:
        raise ValueError("target must be 3D and pred must be 4D.")
    if pred.shape[1:] != target.shape:
        raise ValueError(
            "pred shape must be (num_samples, batch, N, N) matching target."
        )

    # Compute reachability for target once: (B, N, N)
    tgt_reach = _transitive_closure_bool(target)

    # Compute reachability for all predicted samples: (S, B, N, N)
    pred_reach = _transitive_closure_bool(pred)

    # F1 per sample and batch: (S, B)
    tp = (pred_reach & tgt_reach.unsqueeze(0)).sum(dim=(-1, -2)).to(dtype=torch.float32)
    fp = (
        (pred_reach & ~tgt_reach.unsqueeze(0)).sum(dim=(-1, -2)).to(dtype=torch.float32)
    )
    fn = (
        ((~pred_reach) & tgt_reach.unsqueeze(0))
        .sum(dim=(-1, -2))
        .to(dtype=torch.float32)
    )

    denom = 2.0 * tp + fp + fn
    f1 = torch.where(denom > 0, (2.0 * tp) / denom, torch.ones_like(denom))
    return f1.mean(dim=0)


class Metrics(BaseMetrics):
    """
    Unified Metrics handler for graph structure learning (Probabilistic only).

    Takes in different kinds of metrics and handles calculation, accumulation,
    and distributed logic internally.
    """

    def __init__(self, metrics: List[str] | None = None):
        super().__init__()
        self.metrics_list = (
            metrics
            if metrics is not None
            else [
                "e-shd",
                "e-edgef1",
                "e-sid",
                "graph_nll",
                "edge_entropy",
                "ancestor_f1",
                "auc",
            ]
        )

    def _compute_batch_metrics(
        self, targets: torch.Tensor, samples: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute the configured metrics for a single batch.

        Args:
            targets: (B, N, N) ground truth adjacency.
            samples: (S, B, N, N) sampled graphs.
        """
        batch_metrics: Dict[str, float] = {}

        if "e-shd" in self.metrics_list:
            batch_metrics["e-shd"] = float(expected_shd(targets, samples).mean().item())

        if "e-edgef1" in self.metrics_list:
            batch_metrics["e-edgef1"] = float(
                expected_f1_score(targets, samples).mean().item()
            )

        if "ancestor_f1" in self.metrics_list:
            batch_metrics["ancestor_f1"] = float(
                ancestor_f1_score(targets, samples).mean().item()
            )

        if "e-sid" in self.metrics_list:
            batch_metrics["e-sid"] = float(
                structural_interventional_distance(targets, samples).mean().item()
            )

        if "graph_nll" in self.metrics_list:
            batch_metrics["graph_nll"] = graph_nll_score(targets, samples)

        if "edge_entropy" in self.metrics_list:
            batch_metrics["edge_entropy"] = float(edge_entropy(samples).item())

        if "auc" in self.metrics_list:
            batch_metrics["auc"] = float(auc_graph_scores(targets, samples).mean().item())

        return batch_metrics

    def update(
        self,
        targets: torch.Tensor,
        samples: torch.Tensor,
        prefix: str | None = None,
        family: Optional[SCMFamily] = None,
        seeds: Optional[Union[List[int], torch.Tensor]] = None,
        obs_data: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Compute metrics for a batch and update internal history.

        Args:
            targets: (B, N, N) ground truth adjacency.
            samples: (S, B, N, N) binary graph samples.
            prefix: Optional prefix for metric names.
            family: Optional SCMFamily for on-the-fly generation (not used for structural metrics).
            seeds: Optional seeds for the batch.
            obs_data: Optional observational data.
        """
        batch_metrics = self._compute_batch_metrics(targets, samples)

        # Store
        for k, v in batch_metrics.items():
            self.history[k].append(v)
            if prefix:
                self.history[f"{prefix}/{k}"].append(v)

    def compute(
        self,
        probs: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        *,
        samples: Optional[torch.Tensor] = None,
        summary_stats: bool = True,
    ) -> Dict[str, Any]:
        """
        Two usage modes:
        1) Stateful aggregation:
           - call `update(...)` repeatedly
           - call `compute(summary_stats=...)` to summarize accumulated history
        2) One-shot computation:
           - call `compute(probs, targets, samples=...)` to compute batch metrics

        Args:
            probs: Unused placeholder for compatibility with older call sites/tests.
            targets: (B, N, N) ground truth adjacency.
            samples: (S, B, N, N) sampled graphs.
            summary_stats: If True, returns `{metric}_mean/sem/std`. Else returns mean per metric key.
        """
        _ = probs

        if targets is not None and samples is not None:
            # One-shot compute returns scalar metrics (not a history summary).
            # Callers can sync separately if desired.
            return self._compute_batch_metrics(targets, samples)

        # Stateful mode: summarize gathered history.
        full_history = self._gather_history()
        return BaseMetrics._summarize_history(full_history, summary_stats=summary_stats)


def log_prob_graph_scores(
    targets: torch.Tensor, preds: torch.Tensor, eps: float = 1e-6
) -> List[float]:
    """
    Log-probability of each target graph under a Bernoulli distribution with parameters
    given by the mean of sampled graphs.

    Args:
        targets: (B, N, N) binary adjacency.
        preds: (S, B, N, N) binary adjacency samples.
        eps: clamp to avoid log(0).

    Returns:
        List[float] of length B with per-graph log-probabilities.
    """
    if targets.ndim != 3 or preds.ndim != 4:
        raise ValueError("targets must be 3D and preds must be 4D.")
    if preds.shape[1:] != targets.shape:
        raise ValueError("preds shape must be (num_samples, batch, N, N) matching targets.")

    p = preds.float().mean(dim=0).clamp(eps, 1.0 - eps)
    y = targets.float()
    log_prob = (y * torch.log(p) + (1.0 - y) * torch.log(1.0 - p)).sum(dim=(-1, -2))
    return [float(v) for v in log_prob.detach().cpu().tolist()]
