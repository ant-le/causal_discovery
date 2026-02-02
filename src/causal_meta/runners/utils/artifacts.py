from __future__ import annotations

import gzip
import os
from pathlib import Path
from typing import Any, Mapping

import torch


def _cache_cfg(cfg: Any) -> Mapping[str, Any]:
    # Support DictConfig-like objects as well as plain dicts.
    inference = getattr(cfg, "inference", None)
    if inference is None:
        return {}
    if isinstance(inference, Mapping):
        return inference
    return dict(inference)  # type: ignore[arg-type]


def cache_suffix(*, compress: bool) -> str:
    return ".pt.gz" if compress else ".pt"


def cache_settings(cfg: Any) -> tuple[bool, str, int | None]:
    inf = _cache_cfg(cfg)
    compress = bool(inf.get("cache_compress", False))
    dtype = str(inf.get("cache_dtype", "float32")).lower()
    max_samples_raw = inf.get("cache_n_samples", None)
    max_samples = int(max_samples_raw) if max_samples_raw is not None else None
    if max_samples is not None and max_samples < 1:
        max_samples = None
    return compress, dtype, max_samples


def prepare_graph_samples_for_cache(
    samples: torch.Tensor, *, dtype: str, max_samples: int | None
) -> torch.Tensor:
    # samples: (B, K, N, N)
    if samples.ndim != 4:
        raise ValueError("Expected graph samples of shape (B, K, N, N).")

    if max_samples is not None:
        samples = samples[:, :max_samples]

    dtype_norm = dtype.lower()
    if dtype_norm in {"float32", "fp32"}:
        return samples.to(dtype=torch.float32)
    if dtype_norm in {"float16", "fp16"}:
        return samples.to(dtype=torch.float16)
    if dtype_norm in {"bfloat16", "bf16"}:
        return samples.to(dtype=torch.bfloat16)
    if dtype_norm in {"uint8", "u8"}:
        return (samples > 0.5).to(dtype=torch.uint8)
    if dtype_norm in {"bool", "boolean"}:
        return (samples > 0.5).to(dtype=torch.bool)

    raise ValueError(
        f"Unsupported inference.cache_dtype='{dtype}'. "
        "Use one of: float32, float16, bfloat16, uint8, bool."
    )


def atomic_torch_save(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = Path(f"{path}.tmp.{os.getpid()}")
    try:
        if path.suffix == ".gz":
            with gzip.open(tmp_path, "wb") as f:
                torch.save(obj, f)
        else:
            torch.save(obj, tmp_path)
        os.replace(tmp_path, path)
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass


def torch_load(path: Path) -> Any:
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as f:
            return torch.load(f, map_location="cpu")
    return torch.load(path, map_location="cpu")


def find_inference_artifact(
    inference_root: Path,
    *,
    dataset_name: str,
    seed: int,
    prefer_compress: bool,
) -> Path | None:
    preferred = inference_root / dataset_name / f"seed_{seed}{cache_suffix(compress=prefer_compress)}"
    if preferred.exists():
        return preferred
    fallback = inference_root / dataset_name / f"seed_{seed}{cache_suffix(compress=not prefer_compress)}"
    if fallback.exists():
        return fallback
    return None

