from __future__ import annotations

from typing import Any


def infer_explicit_profile(dataset_name: str, family: Any | None) -> str | None:
    """Infer an explicit-baseline profile from dataset metadata.

    Args:
        dataset_name: Name of the evaluation dataset/family.
        family: Optional SCM family object.

    Returns:
        One of ``"linear"``, ``"neuralnet"``, ``"gpcde"``, or ``None``.
    """
    name_l = dataset_name.lower()
    if "linear" in name_l:
        return "linear"
    if "neuralnet" in name_l or "mlp" in name_l:
        return "neuralnet"
    if "gpcde" in name_l:
        return "gpcde"

    if family is None:
        return None

    mechanism_factory = getattr(family, "mechanism_factory", None)
    if mechanism_factory is None:
        return None

    mech_name = mechanism_factory.__class__.__name__.lower()
    if "linear" in mech_name and "mixture" not in mech_name:
        return "linear"
    if "mlp" in mech_name:
        return "neuralnet"
    if "gp" in mech_name:
        return "gpcde"
    return None


def apply_explicit_profile(model: Any, profile: str | None) -> bool:
    """Apply a profile to explicit models that support dynamic overrides.

    Args:
        model: Model instance.
        profile: Profile identifier or ``None``.

    Returns:
        ``True`` if a profile setter was called, else ``False``.
    """
    setter = getattr(model, "set_inference_profile", None)
    if not callable(setter):
        return False
    setter(profile)
    return True
