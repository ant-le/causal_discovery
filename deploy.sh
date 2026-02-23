#!/usr/bin/env bash
# deploy.sh — One-command cluster deploy: pull → sync → submit.
#
# Usage:
#   ./deploy.sh                             # full RQ1 sweep, single A100
#   ./deploy.sh --config smoke_multimodel   # smoke test
#   ./deploy.sh --launcher vsc_a100s        # 4× A100 DDP
#   ./deploy.sh --dry-run                   # print sbatch command without submitting
#   ./deploy.sh --skip-sync                 # skip git pull + uv sync (fast re-submit)
#   ./deploy.sh -- trainer.max_steps=1000   # pass extra Hydra overrides after --
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# ── Defaults ───────────────────────────────────────────────────────────
CONFIG="full_multimodel"
LAUNCHER="vsc_a100"
DRY_RUN=false
SKIP_SYNC=false
HYDRA_EXTRA=()

# ── Parse arguments ───────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)    CONFIG="$2"; shift 2 ;;
    --launcher)  LAUNCHER="$2"; shift 2 ;;
    --dry-run)   DRY_RUN=true; shift ;;
    --skip-sync) SKIP_SYNC=true; shift ;;
    --)          shift; HYDRA_EXTRA=("$@"); break ;;
    -h|--help)
      echo "Usage: ./deploy.sh [--config NAME] [--launcher NAME] [--dry-run] [--skip-sync] [-- HYDRA_OVERRIDES...]"
      echo ""
      echo "Options:"
      echo "  --config NAME    Hydra config name (default: full_multimodel)"
      echo "  --launcher NAME  Launcher preset (default: vsc_a100)"
      echo "  --dry-run        Print the command without submitting"
      echo "  --skip-sync      Skip git pull and uv sync (fast re-submit)"
      echo "  -- OVERRIDES     Extra Hydra overrides passed directly"
      exit 0 ;;
    *)           echo "Unknown arg: $1. Use -- before Hydra overrides."; exit 1 ;;
  esac
done

info() { printf '\033[1;34m==> %s\033[0m\n' "$1"; }
warn() { printf '\033[1;33m==> %s\033[0m\n' "$1"; }

# ── Sync ──────────────────────────────────────────────────────────────
if [[ "$SKIP_SYNC" == false ]]; then
  info "Pulling latest changes"
  git pull --ff-only

  info "Syncing environment (uv)"
  uv sync --extra cluster --extra wandb --frozen --no-editable

  # uv sync installs CPU-only JAX (via dibs-lib). Re-install CUDA JAX
  # so DiBS can use the GPU. This must run after every uv sync.
  JAX_EXTRAS="${CAUSAL_META_JAX_EXTRAS:-cuda12-local}"
  if [[ "$JAX_EXTRAS" != "none" ]]; then
    info "Installing JAX CUDA backend: jax[${JAX_EXTRAS}]"
    uv pip install --python .venv/bin/python --upgrade "jax[${JAX_EXTRAS}]"
  fi

  info "Validating JAX backend:"
  .venv/bin/python -c "
import jax
platforms = sorted({d.platform for d in jax.devices()})
print(f'  jax {jax.__version__}, platforms: {platforms}')
"

  info "Installed causal-meta version:"
  .venv/bin/python -c "import causal_meta; print(f'  {causal_meta.__version__}')"
else
  warn "Skipping git pull + uv sync (--skip-sync)"
fi

# ── Environment ───────────────────────────────────────────────────────
# Prevent eager CUDA init on login node (submitit pickling fix).
export CUDA_VISIBLE_DEVICES=""

# BayesDAG external python (if available).
BAYESDAG_PYTHON="$ROOT_DIR/.venv-bayesdag/bin/python"
if [[ -x "$BAYESDAG_PYTHON" ]]; then
  export CAUSAL_META_BAYESDAG_PYTHON="$BAYESDAG_PYTHON"
  info "BayesDAG python: $BAYESDAG_PYTHON"
else
  warn "BayesDAG env not found — BayesDAG jobs will fail. Run bootstrap_uv.sh first."
fi

# ── Build command ─────────────────────────────────────────────────────
CMD=(
  .venv/bin/python -m causal_meta.main
  --multirun
  --config-name "$CONFIG"
  "hydra/launcher=$LAUNCHER"
)

if [[ ${#HYDRA_EXTRA[@]} -gt 0 ]]; then
  CMD+=("${HYDRA_EXTRA[@]}")
fi

info "Submitting: ${CMD[*]}"

if [[ "$DRY_RUN" == true ]]; then
  warn "Dry run — not submitting."
  exit 0
fi

exec "${CMD[@]}"
