from __future__ import annotations

import inspect
import logging
import os
from typing import Any, Callable, Mapping, Optional, Tuple

import numpy as np
import torch

from causal_meta.models.base import BaseModel
from causal_meta.models.factory import register_model

log = logging.getLogger(__name__)


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
        alpha: float | None = None,
        gamma_z: float | None = None,
        gamma_theta: float | None = None,
        n_particles: int | None = None,
        profile_overrides: Mapping[str, Mapping[str, Any]] | None = None,
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
        self.alpha = alpha
        self.gamma_z = gamma_z
        self.gamma_theta = gamma_theta
        self.n_particles = n_particles

        self._base_profile = {
            "mode": mode,
            "alpha": alpha,
            "gamma_z": gamma_z,
            "gamma_theta": gamma_theta,
            "n_particles": n_particles,
        }
        self._profile_overrides: dict[str, dict[str, Any]] = {}
        if profile_overrides is not None:
            for name, values in profile_overrides.items():
                self._profile_overrides[str(name).lower()] = dict(values)
        self._active_profile: str | None = None
        self._rng_key = None
        self._target_cache = None

    def set_inference_profile(self, profile: str | None) -> None:
        """Apply a named DiBS profile override for explicit comparisons.

        Args:
            profile: Profile identifier (for example ``"linear"``) or ``None``.
        """
        profile_key = str(profile).lower() if profile is not None else "default"
        override = self._profile_overrides.get(profile_key, {})

        prev_state = (
            self.mode,
            self.alpha,
            self.gamma_z,
            self.gamma_theta,
            self.n_particles,
        )

        self.mode = str(override.get("mode", self._base_profile["mode"]))
        self.alpha = self._optional_float(
            override.get("alpha", self._base_profile["alpha"])
        )
        self.gamma_z = self._optional_float(
            override.get("gamma_z", self._base_profile["gamma_z"])
        )
        self.gamma_theta = self._optional_float(
            override.get("gamma_theta", self._base_profile["gamma_theta"])
        )
        self.n_particles = self._optional_int(
            override.get("n_particles", self._base_profile["n_particles"])
        )

        next_state = (
            self.mode,
            self.alpha,
            self.gamma_z,
            self.gamma_theta,
            self.n_particles,
        )
        if next_state != prev_state:
            self._target_cache = None
        self._active_profile = profile_key

    @property
    def needs_pretraining(self) -> bool:
        return False

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Any:
        """
        Forward pass placeholder for explicit inference models.

        Args:
            x: Input data tensor of shape (Batch, Samples, Variables).
            mask: Unused — accepted for interface compatibility.

        Raises:
            RuntimeError: Always raised since DiBS is sampled via `sample`.
        """
        raise RuntimeError("DiBSModel does not implement forward(); use sample().")

    def sample(
        self,
        x: torch.Tensor,
        num_samples: int = 1,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Sample adjacency matrices from the DiBS posterior.

        Args:
            x: Input data tensor of shape (Batch, Samples, Variables).
            num_samples: Number of graph samples to generate per batch element.
            mask: Unused — accepted for interface compatibility.

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

        jax_platforms = sorted({d.platform for d in jax.devices()})
        log.info(
            "DiBS sample: jax_platforms=%s, jax_version=%s, "
            "batch_size=%d, num_nodes=%d, num_samples=%d, "
            "mode=%s, steps=%d, n_particles=%s",
            jax_platforms,
            jax.__version__,
            batch_size,
            num_nodes,
            num_samples,
            self.mode,
            self.steps,
            self.n_particles,
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

            n_particles = (
                int(self.n_particles)
                if self.n_particles is not None
                else int(num_samples)
            )
            if n_particles < 1:
                raise ValueError("DiBS n_particles must be >= 1.")

            graphs, _ = dibs.sample(
                key=sample_key,
                n_particles=n_particles,
                steps=int(self.steps),
            )

            graphs_np = np.asarray(graphs)
            graphs_t = torch.from_numpy(graphs_np.copy()).to(device=x.device)
            graphs_t = (graphs_t > 0.5).to(dtype=torch.float32)
            graphs_t = self._match_num_samples(
                graphs_t,
                target_samples=int(num_samples),
                seed=self.seed + batch_idx,
            )
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
            result = make_target(
                key=key,
                n_vars=self.num_nodes,
                **self._build_target_kwargs(make_target),
            )
            if isinstance(result, tuple) and len(result) == 3:
                _, graph_model, likelihood_model = result
            else:
                graph_model, likelihood_model = result
            self._target_cache = (graph_model, likelihood_model)
        return self._target_cache

    def _build_target_kwargs(
        self,
        make_target: Callable[..., Tuple[Any, Any]],
    ) -> dict[str, float]:
        """Build optional target kwargs supported by the installed DiBS version."""
        params = set(inspect.signature(make_target).parameters.keys())
        kwargs: dict[str, float] = {}

        self._set_first_supported_param(
            kwargs,
            params,
            candidates=("alpha_linear", "alpha"),
            value=self.alpha,
        )
        self._set_first_supported_param(
            kwargs,
            params,
            candidates=("gamma_z", "gamma_latent"),
            value=self.gamma_z,
        )
        self._set_first_supported_param(
            kwargs,
            params,
            candidates=("gamma_theta", "gamma"),
            value=self.gamma_theta,
        )

        return kwargs

    @staticmethod
    def _set_first_supported_param(
        kwargs: dict[str, float],
        supported: set[str],
        *,
        candidates: tuple[str, ...],
        value: float | None,
    ) -> None:
        if value is None:
            return
        for name in candidates:
            if name in supported:
                kwargs[name] = float(value)
                return

    @staticmethod
    def _match_num_samples(
        graphs_t: torch.Tensor,
        *,
        target_samples: int,
        seed: int,
    ) -> torch.Tensor:
        """Ensure DiBS output has exactly ``target_samples`` graph draws."""
        if target_samples < 1:
            raise ValueError("target_samples must be >= 1.")
        available = int(graphs_t.shape[0])
        if available == target_samples:
            return graphs_t
        if available > target_samples:
            return graphs_t[:target_samples]

        generator = torch.Generator(device=graphs_t.device)
        generator.manual_seed(int(seed))
        indices = torch.randint(
            low=0,
            high=available,
            size=(target_samples,),
            device=graphs_t.device,
            generator=generator,
        )
        return graphs_t.index_select(0, indices)

    @staticmethod
    def _optional_float(value: Any) -> float | None:
        if value is None:
            return None
        return float(value)

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        if value is None:
            return None
        return int(value)

    def _require_dibs(self) -> Tuple[Any, Any, Any, Callable[..., Tuple[Any, Any]]]:
        if not self.xla_preallocate:
            os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

        try:
            import jax
            import jax.numpy as jnp
            from dibs.inference import JointDiBS, MarginalDiBS
            from dibs.target import (
                make_linear_gaussian_model,
                make_nonlinear_gaussian_model,
            )
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
