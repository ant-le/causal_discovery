#!/usr/bin/env bash

resolve_main_venv_dir() {
  local root_dir="$1"
  local venv_dir="${VENV_DIR:-${UV_PROJECT_ENVIRONMENT:-.venv}}"
  if [[ "${venv_dir}" != /* ]]; then
    venv_dir="${root_dir}/${venv_dir}"
  fi
  printf '%s\n' "${venv_dir}"
}

load_bootstrap_env_if_available() {
  local root_dir="$1"
  local env_snippet="${root_dir}/.bootstrap_env.sh"
  if [[ -f "${env_snippet}" ]]; then
    # shellcheck disable=SC1090
    source "${env_snippet}"
  fi
}

require_main_env_bins() {
  local context="$1"
  local main_python="$2"
  local required_bin="$3"
  if [[ ! -x "${main_python}" ]] || [[ ! -x "${required_bin}" ]]; then
    echo "[${context}] ERROR: main environment is missing." >&2
    echo "[${context}] Expected python at: ${main_python}" >&2
    echo "[${context}] Expected executable at: ${required_bin}" >&2
    echo "[${context}] Run: uv sync --extra cluster --extra wandb --frozen --no-editable" >&2
    echo "[${context}] Or set VENV_DIR=/path/to/venv (or UV_PROJECT_ENVIRONMENT)." >&2
    exit 1
  fi
}

require_main_torch_cuda() {
  local context="$1"
  local main_python="$2"
  if ! "${main_python}" -c "import torch" >/dev/null 2>&1; then
    echo "[${context}] ERROR: torch is not installed in main environment." >&2
    echo "[${context}] Run: uv sync --extra cluster --extra wandb --frozen --no-editable" >&2
    exit 1
  fi

  if ! "${main_python}" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" >/dev/null 2>&1; then
    echo "[${context}] ERROR: CUDA is not available in main environment torch." >&2
    echo "[${context}] This workflow supports GPU-only runs by default." >&2
    exit 1
  fi
}

ensure_bayesdag_python() {
  local root_dir="$1"
  if [[ -n "${CAUSAL_META_BAYESDAG_PYTHON:-}" ]]; then
    return
  fi

  local default_bayesdag_python="${root_dir}/.venv-bayesdag/bin/python"
  if [[ -x "${default_bayesdag_python}" ]]; then
    export CAUSAL_META_BAYESDAG_PYTHON="${default_bayesdag_python}"
    return
  fi

  echo "CAUSAL_META_BAYESDAG_PYTHON is required for model=bayesdag." >&2
  echo "Run ./bootstrap_uv.sh or export CAUSAL_META_BAYESDAG_PYTHON manually." >&2
  exit 1
}

require_bayesdag_torch_cuda() {
  local context="$1"
  local bayesdag_python="$2"
  if ! "${bayesdag_python}" -c "import causica, torch" >/dev/null 2>&1; then
    echo "[${context}] ERROR: BayesDAG environment is incomplete (missing causica and/or torch)." >&2
    echo "[${context}] Run: ./bootstrap_uv.sh" >&2
    exit 1
  fi

  if ! "${bayesdag_python}" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" >/dev/null 2>&1; then
    echo "[${context}] ERROR: CUDA is not available in BayesDAG torch environment." >&2
    echo "[${context}] This workflow supports GPU-only runs by default." >&2
    exit 1
  fi
}

require_dibs_jax_gpu() {
  local context="$1"
  local main_python="$2"
  if ! "${main_python}" -c "import jax, sys; platforms={d.platform for d in jax.devices()}; sys.exit(0 if ('gpu' in platforms or 'cuda' in platforms) else 1)" >/dev/null 2>&1; then
    echo "[${context}] ERROR: JAX GPU backend is not available for DiBS." >&2
    echo "[${context}] Install CUDA JAX in main env (for example: jax[cuda12-local])." >&2
    exit 1
  fi
}
