from __future__ import annotations

import re
from typing import Any

# ── Known tokens used in dg_2pretrain task names ────────────────────────
# Mechanisms as they appear in task names, mapped to the canonical profile
# name used in model profile_overrides.  Multi-word mechanisms (e.g.
# ``logistic_map``) must appear before any single-word prefix that could
# match a substring, so the list is ordered longest-first.
_MECHANISM_TO_PROFILE: dict[str, str] = {
    "logistic_map": "logistic",
    "pnl_tanh": "pnl",
    "neuralnet": "neuralnet",
    "periodic": "periodic",
    "linear": "linear",
    "square": "square",
    "gpcde": "gpcde",
}

# Graph tokens as they appear in task names, mapped to the profile suffix.
# Sparse baselines (er20, sf1) map to "" (no suffix) because the base
# mechanism profile is already tuned for sparse graphs.
_GRAPH_TO_SUFFIX: dict[str, str] = {
    "er20": "",
    "er40": "_er40",
    "er60": "_er60",
    "sf1": "",
    "sf2": "_sf2",
    "sf3": "_sf3",
    "sbm": "_sbm",
    "ws": "_ws",
    "grg": "_grg",
}

# Legacy set kept for ``strip_graph_suffix`` (profile-key → mechanism).
_GRAPH_SUFFIXES = frozenset(s for s in _GRAPH_TO_SUFFIX.values() if s)

# OOD mechanisms and their closest ID mechanism for density-fallback.
_OOD_MECHANISM_FALLBACK: dict[str, str] = {
    "periodic": "gpcde",
    "square": "gpcde",
    "logistic": "gpcde",
    "pnl": "gpcde",
}


def strip_graph_suffix(profile_key: str) -> str:
    """Strip a known graph-topology suffix from a profile key.

    Returns the mechanism-only part if a known suffix is found,
    otherwise returns *profile_key* unchanged.
    """
    for suffix in _GRAPH_SUFFIXES:
        if profile_key.endswith(suffix):
            return profile_key[: -len(suffix)]
    return profile_key


def compute_fallback_keys(profile_key: str) -> list[str]:
    """Return profile keys to try, in priority order.

    The lookup chain is:

    1. Exact key (e.g. ``"logistic_er40"``).
    2. Mechanism-only (e.g. ``"logistic"``).
    3. Closest ID mechanism + graph suffix (e.g. ``"gpcde_er40"``).
    4. Closest ID mechanism base (e.g. ``"gpcde"``).

    Steps 3-4 only apply when the mechanism is an OOD mechanism with a
    known ID fallback (see :data:`_OOD_MECHANISM_FALLBACK`).
    """
    mech = strip_graph_suffix(profile_key)
    suffix = profile_key[len(mech) :]

    keys: list[str] = [profile_key]

    # 2. mechanism-only (only if there was a suffix)
    if mech != profile_key:
        keys.append(mech)

    # 3-4. ID-mechanism fallback for OOD mechanisms
    mapped = _OOD_MECHANISM_FALLBACK.get(mech)
    if mapped is not None:
        if suffix:
            keys.append(mapped + suffix)
        keys.append(mapped)

    return keys


def _extract_graph_suffix(name_l: str) -> str:
    """Extract graph-topology suffix from a dataset name for profile matching.

    Uses the canonical :data:`_GRAPH_TO_SUFFIX` mapping.  Returns the
    profile suffix (e.g. ``"_er40"``, ``"_sbm"``) or ``""`` for sparse
    baseline graphs (er20, sf1) and unknown graph types.
    """
    for token, suffix in _GRAPH_TO_SUFFIX.items():
        # Look for the graph token bounded by underscores (or at the start
        # of the string) to avoid false substring hits.
        if re.search(rf"(?:^|_){re.escape(token)}(?:_|$)", name_l):
            return suffix
    return ""


def _extract_mechanism(name_l: str) -> str | None:
    """Extract mechanism profile name from a dataset name.

    Iterates :data:`_MECHANISM_TO_PROFILE` (longest tokens first) and
    returns the canonical profile name for the first match, or ``None``.
    """
    for token, profile in _MECHANISM_TO_PROFILE.items():
        if token in name_l:
            return profile
    return None


def infer_explicit_profile(dataset_name: str, family: Any | None) -> str | None:
    """Infer an explicit-baseline profile from dataset metadata.

    Extracts the mechanism type and graph topology from the structured
    *dataset_name* (as defined in ``dg_2pretrain_multimodel.yaml``).  The
    returned profile key can be passed to :func:`compute_fallback_keys`
    to resolve to the best matching entry in a model's
    ``profile_overrides`` dict.

    Args:
        dataset_name: Name of the evaluation dataset/family
            (e.g. ``"id_linear_er40_d20_n500"``).
        family: Optional SCM family object used as a last resort when
            the mechanism cannot be inferred from the name alone.

    Returns:
        Profile identifier with mechanism and optional graph-type suffix
        (e.g., ``"linear_sf2"``, ``"logistic_er40"``), or ``None``.
    """
    name_l = dataset_name.lower()

    # Extract mechanism type from the structured task name.
    mech = _extract_mechanism(name_l)

    # Fallback: try to infer from the family object's mechanism factory.
    if mech is None:
        if family is None:
            return None
        mechanism_factory = getattr(family, "mechanism_factory", None)
        if mechanism_factory is None:
            return None
        mech_name = mechanism_factory.__class__.__name__.lower()
        if "linear" in mech_name and "mixture" not in mech_name:
            mech = "linear"
        elif "mlp" in mech_name:
            mech = "neuralnet"
        elif "gp" in mech_name:
            mech = "gpcde"

    if mech is None:
        return None

    # Extract graph topology suffix.
    graph_suffix = _extract_graph_suffix(name_l)

    return mech + graph_suffix


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
