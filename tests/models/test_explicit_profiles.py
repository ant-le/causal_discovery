from __future__ import annotations

from types import SimpleNamespace

from causal_meta.models.bayesdag.model import BayesDAGModel
from causal_meta.models.dibs.model import DiBSModel
from causal_meta.runners.utils.explicit_profiles import (
    apply_explicit_profile,
    infer_explicit_profile,
)


def test_infer_explicit_profile_from_dataset_name() -> None:
    assert infer_explicit_profile("id_linear_er20", None) == "linear"
    assert infer_explicit_profile("id_neuralnet_er40", None) == "neuralnet"
    assert infer_explicit_profile("id_gpcde_er60", None) == "gpcde"


def test_infer_explicit_profile_from_family_mechanism() -> None:
    class GPMechanismFactory:
        pass

    family = SimpleNamespace(mechanism_factory=GPMechanismFactory())
    assert infer_explicit_profile("ood_mech_periodic_er40", family) == "gpcde"


def test_apply_explicit_profile_calls_setter() -> None:
    class DummyModel:
        def __init__(self) -> None:
            self.profile = None

        def set_inference_profile(self, profile: str | None) -> None:
            self.profile = profile

    model = DummyModel()
    assert apply_explicit_profile(model, "linear") is True
    assert model.profile == "linear"


def test_dibs_profile_override_updates_hyperparameters() -> None:
    model = DiBSModel(
        num_nodes=5,
        mode="nonlinear",
        alpha=0.02,
        gamma_z=5.0,
        gamma_theta=1000.0,
        n_particles=32,
        profile_overrides={
            "linear": {
                "mode": "linear",
                "alpha": 0.2,
                "gamma_z": 5.0,
                "gamma_theta": 500.0,
                "n_particles": 64,
            }
        },
    )

    model.set_inference_profile("linear")

    assert model.mode == "linear"
    assert model.alpha == 0.2
    assert model.gamma_z == 5.0
    assert model.gamma_theta == 500.0
    assert model.n_particles == 64


def test_bayesdag_profile_override_updates_hyperparameters() -> None:
    model = BayesDAGModel(
        num_nodes=5,
        variant="nonlinear",
        lambda_sparse=10.0,
        num_chains=10,
        scale_noise=0.001,
        scale_noise_p=0.01,
        profile_overrides={
            "gpcde": {
                "variant": "nonlinear",
                "lambda_sparse": 1.0,
                "num_chains": 10,
                "scale_noise": 0.1,
                "scale_noise_p": 0.001,
            }
        },
    )

    model.set_inference_profile("gpcde")

    assert model.variant == "nonlinear"
    assert model.lambda_sparse == 1.0
    assert model.num_chains == 10
    assert model.scale_noise == 0.1
    assert model.scale_noise_p == 0.001
