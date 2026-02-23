#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

MAIN_PYTHON_VERSION="${CAUSAL_META_MAIN_PYTHON_VERSION:-3.11}"
BAYESDAG_PYTHON_VERSION="${CAUSAL_META_BAYESDAG_PYTHON_VERSION:-3.9}"
JAX_EXTRAS="${CAUSAL_META_JAX_EXTRAS:-cuda12-local}"
STRICT_CUDA_JAX="${CAUSAL_META_STRICT_CUDA_JAX:-0}"
BAYESDAG_REPO_URL="${CAUSAL_META_BAYESDAG_REPO_URL:-https://github.com/microsoft/Project-BayesDAG.git}"
BAYESDAG_REPO_REF="${CAUSAL_META_BAYESDAG_REPO_REF:-}"

MAIN_VENV="$ROOT_DIR/.venv"
BAYESDAG_VENV="$ROOT_DIR/.venv-bayesdag"
MAIN_PYTHON="$MAIN_VENV/bin/python"
BAYESDAG_PYTHON="$BAYESDAG_VENV/bin/python"
ENV_SNIPPET="$ROOT_DIR/.bootstrap_env.sh"

info() {
  printf '==> %s\n' "$1"
}

die() {
  printf 'Error: %s\n' "$1" >&2
  exit 1
}

if [[ ! -f "pyproject.toml" ]]; then
  die "run this script from the repository root."
fi

if ! command -v uv >/dev/null 2>&1; then
  die "uv is not installed. Install it first: https://docs.astral.sh/uv/getting-started/installation/"
fi

info "Installing Python runtimes with uv"
uv python install "$MAIN_PYTHON_VERSION" "$BAYESDAG_PYTHON_VERSION"

info "Creating main environment (.venv, Python ${MAIN_PYTHON_VERSION})"
uv venv "$MAIN_VENV" --python "$MAIN_PYTHON_VERSION" --clear
uv sync --python "$MAIN_PYTHON" --extra cluster --extra wandb --frozen
uv pip install --python "$MAIN_PYTHON" --no-deps --editable "$ROOT_DIR"

if [[ "$JAX_EXTRAS" != "none" ]]; then
  info "Installing JAX backend in .venv: jax[${JAX_EXTRAS}]"
  uv pip install --python "$MAIN_PYTHON" --upgrade "jax[${JAX_EXTRAS}]"

  INSTALLED_JAX_VERSION="$($MAIN_PYTHON - <<'PY'
from importlib import metadata

try:
    print(metadata.version("jax"))
except metadata.PackageNotFoundError:
    print("")
PY
)"
  INSTALLED_JAXLIB_VERSION="$($MAIN_PYTHON - <<'PY'
from importlib import metadata

try:
    print(metadata.version("jaxlib"))
except metadata.PackageNotFoundError:
    print("")
PY
)"

  if [[ -n "$INSTALLED_JAX_VERSION" && -n "$INSTALLED_JAXLIB_VERSION" && "$INSTALLED_JAX_VERSION" != "$INSTALLED_JAXLIB_VERSION" ]]; then
    info "Aligning jaxlib to jax version ${INSTALLED_JAX_VERSION} (found ${INSTALLED_JAXLIB_VERSION})"
    if ! uv pip install --python "$MAIN_PYTHON" --upgrade "jaxlib==${INSTALLED_JAX_VERSION}"; then
      info "Falling back to jax==${INSTALLED_JAXLIB_VERSION} to match available jaxlib"
      uv pip install --python "$MAIN_PYTHON" --upgrade "jax==${INSTALLED_JAXLIB_VERSION}"
    fi
  fi
fi

info "Validating active causal_meta package in .venv"
CAUSAL_META_BOOTSTRAP_ROOT="$ROOT_DIR" "$MAIN_PYTHON" - <<'PY'
import inspect
import os
from pathlib import Path

import causal_meta
from causal_meta.models.utils.nn import CausalAdjacencyMatrix

root = Path(os.environ["CAUSAL_META_BOOTSTRAP_ROOT"]).resolve()
module_path = Path(causal_meta.__file__).resolve()
if root not in module_path.parents:
    raise SystemExit(
        "causal_meta is not loaded from this checkout: "
        f"{module_path}"
    )

smoke_cfg = (module_path.parent / "configs" / "smoke_multimodel.yaml").read_text()
if "logger: wandb" not in smoke_cfg:
    raise SystemExit(
        "smoke_multimodel.yaml in active package is stale; expected logger: wandb"
    )

forward_src = inspect.getsource(CausalAdjacencyMatrix.forward)
if ".view(tgt_len, bsz * self.num_heads, head_dim)" in forward_src:
    raise SystemExit(
        "active causal_meta package still contains stale BCNP .view() projection code"
    )

print(f"causal_meta import path: {module_path}")
PY

if [[ "$JAX_EXTRAS" != "none" ]]; then
  info "Validating JAX backend"
  CAUSAL_META_BOOTSTRAP_JAX_EXTRAS="$JAX_EXTRAS" \
    CAUSAL_META_STRICT_CUDA_JAX="$STRICT_CUDA_JAX" \
    "$MAIN_PYTHON" - <<'PY'
from importlib import util
import os

import jax

extras = os.environ["CAUSAL_META_BOOTSTRAP_JAX_EXTRAS"]
platforms = sorted({device.platform for device in jax.devices()})
platform_str = ", ".join(platforms) if platforms else "none"
print(f"jax version: {jax.__version__}")
print(f"jax platforms: {platform_str}")

if extras.startswith("cuda"):
    plugin_prefix = extras.split("-", maxsplit=1)[0].replace("-", "_")
    plugin_module = f"jax_{plugin_prefix}_plugin"
    if util.find_spec(plugin_module) is None:
        print(
            f"WARNING: expected CUDA plugin module '{plugin_module}' was not found."
        )

if os.environ.get("CAUSAL_META_STRICT_CUDA_JAX") == "1":
    if extras.startswith("cuda") and not any(p in {"gpu", "cuda"} for p in platforms):
        raise SystemExit(
            "JAX CUDA backend is unavailable while CAUSAL_META_STRICT_CUDA_JAX=1"
        )
PY
fi

info "Creating BayesDAG environment (.venv-bayesdag, Python ${BAYESDAG_PYTHON_VERSION})"
uv venv "$BAYESDAG_VENV" --python "$BAYESDAG_PYTHON_VERSION" --clear
uv pip install --python "$BAYESDAG_PYTHON" -r requirements-bayesdag.txt
uv pip install --python "$BAYESDAG_PYTHON" "wandb>=0.15.0"

info "Installing Project-BayesDAG (causica) into .venv-bayesdag"
tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT
git clone --depth 1 "$BAYESDAG_REPO_URL" "$tmp_dir/Project-BayesDAG"
if [[ -n "$BAYESDAG_REPO_REF" ]]; then
  git -C "$tmp_dir/Project-BayesDAG" checkout "$BAYESDAG_REPO_REF"
fi
cp "$tmp_dir/Project-BayesDAG/README.md" "$tmp_dir/Project-BayesDAG/src/README.md"
uv pip install --python "$BAYESDAG_PYTHON" --no-deps "$tmp_dir/Project-BayesDAG/src"

info "Validating BayesDAG environment"
"$BAYESDAG_PYTHON" - <<'PY'
import causica
import torch

print(f"causica import path: {causica.__file__}")
print(f"BayesDAG torch version: {torch.__version__}")
PY

cat > "$ENV_SNIPPET" <<EOF
export PATH="$MAIN_VENV/bin:\$PATH"
export CAUSAL_META_BAYESDAG_PYTHON="$BAYESDAG_PYTHON"
EOF

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  export CAUSAL_META_BAYESDAG_PYTHON="$BAYESDAG_PYTHON"
fi

echo
echo "Setup complete."
echo "Main env python:      $MAIN_PYTHON"
echo "BayesDAG env python:  $BAYESDAG_PYTHON"
echo "Environment snippet:  $ENV_SNIPPET"
echo
echo "Use this before runs that include BayesDAG (or source the snippet):"
echo "  source \"$ENV_SNIPPET\""
echo "DiBS JAX backend defaults to CUDA 12 via jax[cuda12-local]."
echo "Override with (example): CAUSAL_META_JAX_EXTRAS=cuda13 ./bootstrap_uv.sh"
echo "Disable JAX extras install: CAUSAL_META_JAX_EXTRAS=none ./bootstrap_uv.sh"
echo "Require CUDA-visible JAX at setup time: CAUSAL_META_STRICT_CUDA_JAX=1 ./bootstrap_uv.sh"
echo
echo "Run local smoke test:"
echo "  $MAIN_PYTHON -m causal_meta.main name=smoke_test"
