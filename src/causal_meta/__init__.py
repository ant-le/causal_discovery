from __future__ import annotations

import sys
import types
from importlib.metadata import PackageNotFoundError, version


def _maybe_disable_torch_compile_optimizer_wrapper() -> None:
    """Bypass torch._compile optimizer wrappers when optional deps are absent.

    Some local CPU/macOS environments ship a PyTorch build where constructing
    optimizers imports ``torch._dynamo``, which in turn requires ``optree._C``.
    When that optional extension is unavailable, even ``torch.optim.AdamW``
    construction fails.  The underlying optimizer implementation is still
    functional, so we fall back to the undecorated ``add_param_group`` method.
    """

    try:
        import optree._C  # type: ignore[import-not-found]  # noqa: F401
    except Exception:
        try:
            import torch

            fake_dynamo = types.ModuleType("torch._dynamo")

            def _disable(fn=None, recursive=True, wrapping=False):
                _ = recursive, wrapping
                if fn is None:
                    return lambda inner: inner
                return fn

            def _graph_break() -> None:
                return None

            fake_dynamo.disable = _disable
            fake_dynamo.graph_break = _graph_break

            sys.modules.setdefault("torch._dynamo", fake_dynamo)
            torch._dynamo = fake_dynamo
        except Exception:
            pass


_maybe_disable_torch_compile_optimizer_wrapper()

try:
    __version__ = version("causal_meta")
except PackageNotFoundError:
    __version__ = "0.0.0-unknown"
