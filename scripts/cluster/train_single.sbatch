#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
VENV_DIR="${VENV_DIR:-${UV_PROJECT_ENVIRONMENT:-.venv}}"
if [[ "${VENV_DIR}" != /* ]]; then
  VENV_DIR="${ROOT_DIR}/${VENV_DIR}"
fi
MAIN_PYTHON="${VENV_DIR}/bin/python"
CAUSAL_META_BIN="${VENV_DIR}/bin/causal-meta"

MODEL="${1:?Usage: train_single.sbatch <model> [config_name] [run_name] [hydra_overrides...] }"
CONFIG_NAME="${2:-full_multimodel}"
RUN_NAME="${3:-rq1_${MODEL}_single}"

if [[ "$#" -ge 3 ]]; then
  shift 3
else
  shift "$#"
fi

if [[ "${MODEL}" == "bayesdag" ]]; then
  if [[ -z "${CAUSAL_META_BAYESDAG_PYTHON:-}" ]]; then
    export CAUSAL_META_BAYESDAG_PYTHON="${ROOT_DIR}/.venv-bayesdag/bin/python"
  fi
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
export HYDRA_FULL_ERROR=1
export PYTHONFAULTHANDLER=1

"${CAUSAL_META_BIN}" \
  --config-name "${CONFIG_NAME}" \
  "model=${MODEL}" \
  "name=${RUN_NAME}" \
  "$@"
