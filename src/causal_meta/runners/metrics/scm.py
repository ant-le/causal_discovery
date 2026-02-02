from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

from causal_meta.datasets.scm import SCMFamily
from causal_meta.runners.utils.scoring import LinearGaussianScorer

from .base import BaseMetrics

log = logging.getLogger(__name__)


class SCMMetrics(BaseMetrics):
    """
    Metrics handler for SCM-based evaluation (Likelihoods, I-NIL).
    Maintains state across batches and handles distributed synchronization.

    Supports two modes:
    1. Pre-generated: Pass `interventional_data` list.
    2. On-the-fly: Pass `family` and `seeds`.
    """

    def __init__(self, metrics: List[str] | None = None):
        super().__init__()
        self.metrics_list = metrics if metrics is not None else ["inil"]
        self._warned_linear_gaussian = False

    def update(
        self,
        obs_data: torch.Tensor,
        graph_samples: torch.Tensor,
        interventional_data: List[Dict[str, Any]] | None = None,
        prefix: str | None = None,
        family: Optional[SCMFamily] = None,
        seeds: Optional[Union[List[int], torch.Tensor]] = None,
    ):
        """
        Update metrics for a single task (batch of samples).
        """
        if "inil" not in self.metrics_list:
            return

        current_seeds = []
        if seeds is not None:
            current_seeds = (
                seeds.tolist() if isinstance(seeds, torch.Tensor) else list(seeds)
            )

        # We assume update is called per-task (batch_size=1) in the current pipeline evaluation loop.
        # Case 1: On-the-fly generation
        if interventional_data is None and family is not None and current_seeds:
            if not self._warned_linear_gaussian:
                log.warning(
                    "I-NIL currently uses a Linear Gaussian scorer. Treat as a heuristic."
                )
                self._warned_linear_gaussian = True

            seed = int(current_seeds[0])
            # Use base class helper
            interventional_data = self._generate_interventional_data(
                family, seed, n_samples=obs_data.shape[0]
            )
            self._compute_inil_score(
                obs_data, graph_samples, interventional_data, prefix
            )

        # Case 2: Pre-generated data
        elif interventional_data is not None:
            if not self._warned_linear_gaussian:
                log.warning("I-NIL using Linear Gaussian scorer.")
                self._warned_linear_gaussian = True
            self._compute_inil_score(
                obs_data, graph_samples, interventional_data, prefix
            )

    def _compute_inil_score(
        self,
        obs_data: torch.Tensor,
        graph_samples: torch.Tensor,
        interventional_data: List[Dict[str, Any]],
        prefix: str | None,
    ):

        n_graph_samples = int(graph_samples.shape[0])
        task_nlls: List[float] = []

        # Deduplicate graphs for efficiency
        flat = graph_samples.detach().to(dtype=torch.int8).reshape(n_graph_samples, -1)
        unique_flat, inverse = torch.unique(flat, dim=0, return_inverse=True)

        per_unique_nll: List[float] = [float("inf")] * int(unique_flat.shape[0])

        for u in range(int(unique_flat.shape[0])):
            adj = (
                unique_flat[u]
                .view(graph_samples.shape[1], graph_samples.shape[2])
                .to(dtype=graph_samples.dtype, device=graph_samples.device)
            )

            scorer = LinearGaussianScorer(adj, obs_data)
            try:
                scorer.fit()
            except RuntimeError:
                per_unique_nll[u] = float("inf")
                continue

            total_nll_k = 0.0
            total_samples = 0

            for item in interventional_data:
                target = item["target"]
                val = item["value"]
                int_data_x = item["data"].to(obs_data.device)

                nll_avg = scorer.score_nll(
                    int_data_x, intervention_target=target, intervention_value=val
                )
                total_nll_k += nll_avg * int_data_x.shape[0]
                total_samples += int_data_x.shape[0]

            if total_samples > 0:
                per_unique_nll[u] = total_nll_k / total_samples
            else:
                per_unique_nll[u] = 0.0

        for k in range(n_graph_samples):
            task_nlls.append(per_unique_nll[int(inverse[k].item())])

        # Aggregate: - log (mean ( exp ( - nlls ) ) )
        if task_nlls:
            nlls_t = torch.tensor(task_nlls)
            valid_nlls = nlls_t[torch.isfinite(nlls_t)]
            if valid_nlls.numel() > 0:
                neg_nlls = -valid_nlls
                log_sum_exp = torch.logsumexp(neg_nlls, dim=0)
                inil_score = -(log_sum_exp - np.log(len(task_nlls)))

                self.history["inil"].append(inil_score.item())
                if prefix:
                    self.history[f"{prefix}/inil"].append(inil_score.item())

    def compute(self, summary_stats: bool = True) -> Dict[str, Any]:
        full_history = self._gather_history()
        return BaseMetrics._summarize_history(full_history, summary_stats=summary_stats)

    def get_raw_results(self) -> Dict[str, List[float]]:
        return super().get_raw_results()
