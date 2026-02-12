from __future__ import annotations

import os
from typing import Any, Callable, Tuple

import numpy as np
import torch

from causal_meta.models.base import BaseModel
from causal_meta.models.factory import register_model


@register_model("dibs")
class DiBSModel(BaseModel):
    """
    Wrapper for DiBS (Differentiable Bayesian Structure Learning).

    This model runs explicit posterior inference per dataset instance and
    returns graph samples compatible with the inference cache pipeline.
    """

    def __init__(
        self,
        num_nodes: int,
        *,
        mode: str = "nonlinear",
        steps: int = 1000,
        seed: int = 0,
        use_marginal: bool = False,
        xla_preallocate: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        _ = kwargs
        self.num_nodes = num_nodes
        self.mode = mode
        self.steps = steps
        self.seed = seed
        self.use_marginal = use_marginal
        self.xla_preallocate = xla_preallocate
        self._rng_key = None
        self._target_cache = None

    @property
    def needs_pretraining(self) -> bool:
        return False

    def forward(self, x: torch.Tensor) -> Any:
        """
        Forward pass placeholder for explicit inference models.

        Args:
            x: Input data tensor of shape (Batch, Samples, Variables).

        Raises:
            RuntimeError: Always raised since DiBS is sampled via `sample`.
        """
        raise RuntimeError("DiBSModel does not implement forward(); use sample().")

    def sample(self, x: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """
        Sample adjacency matrices from the DiBS posterior.

        Args:
            x: Input data tensor of shape (Batch, Samples, Variables).
            num_samples: Number of graph samples to generate per batch element.

        Returns:
            Sampled adjacency matrices of shape (Batch, num_samples, Variables, Variables).
        """
        jax, jnp, dibs_cls, make_target = self._require_dibs()

        if x.ndim != 3:
            raise ValueError("Input data must have shape (Batch, Samples, Variables).")

        batch_size, _, num_nodes = x.shape
        if num_nodes != self.num_nodes:
            raise ValueError(
                "Input data node count does not match configured num_nodes."
            )

        if self._rng_key is None:
            self._rng_key = jax.random.PRNGKey(self.seed)

        samples_per_batch = []
        for batch_idx in range(batch_size):
            x_np = x[batch_idx].detach().cpu().numpy().astype(np.float32, copy=False)
            x_jax = jnp.asarray(x_np)

            self._rng_key, model_key, sample_key = jax.random.split(self._rng_key, 3)
            graph_model, likelihood_model = self._get_target_models(
                make_target=make_target, key=model_key
            )

            dibs = dibs_cls(
                x=x_jax,
                interv_mask=None,
                graph_model=graph_model,
                likelihood_model=likelihood_model,
            )

            graphs, _ = dibs.sample(
                key=sample_key,
                n_particles=int(num_samples),
                steps=int(self.steps),
            )

            graphs_np = np.asarray(graphs)
            graphs_t = torch.from_numpy(graphs_np.copy()).to(device=x.device)
            graphs_t = (graphs_t > 0.5).to(dtype=torch.float32)
            samples_per_batch.append(graphs_t)

        return torch.stack(samples_per_batch, dim=0)

    def calculate_loss(
        self, output: Any, target: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """
        Loss placeholder for explicit inference models.

        Args:
            output: Model output (unused).
            target: Ground truth adjacency matrix (unused).

        Raises:
            RuntimeError: Always raised since DiBS does not expose a training loss here.
        """
        _ = output
        _ = target
        _ = kwargs
        raise RuntimeError("DiBSModel does not implement calculate_loss().")

    def _get_target_models(
        self, make_target: Callable[..., Tuple[Any, Any]], key: Any
    ) -> Tuple[Any, Any]:
        if self._target_cache is None:
            if self.mode not in {"linear", "nonlinear"}:
                raise ValueError("DiBS mode must be 'linear' or 'nonlinear'.")
            result = make_target(key=key, n_vars=self.num_nodes)
            if isinstance(result, tuple) and len(result) == 3:
                _, graph_model, likelihood_model = result
            else:
                graph_model, likelihood_model = result
            self._target_cache = (graph_model, likelihood_model)
        return self._target_cache

    def _require_dibs(self) -> Tuple[Any, Any, Any, Callable[..., Tuple[Any, Any]]]:
        if not self.xla_preallocate:
            os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

        try:
            import jax
            import jax.numpy as jnp
            from dibs.inference import JointDiBS, MarginalDiBS
            from dibs.target import (make_linear_gaussian_model,
                                     make_nonlinear_gaussian_model)
        except Exception as exc:  # pragma: no cover - exercised via tests
            raise RuntimeError(
                "DiBS requires the 'dibs-lib' package and JAX. "
                "Install with `pip install dibs-lib` and ensure a compatible JAX backend."
            ) from exc

        dibs_cls = MarginalDiBS if self.use_marginal else JointDiBS
        make_target = (
            make_linear_gaussian_model
            if self.mode == "linear"
            else make_nonlinear_gaussian_model
        )

        return jax, jnp, dibs_cls, make_target
