#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
source "${ROOT_DIR}/scripts/cluster/lib.sh"
VENV_DIR="$(resolve_main_venv_dir "${ROOT_DIR}")"
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

require_main_env_bins "train_single" "${MAIN_PYTHON}" "${CAUSAL_META_BIN}"
require_main_torch_cuda "train_single" "${MAIN_PYTHON}"

if [[ "${MODEL}" == "bayesdag" ]]; then
  ensure_bayesdag_python "${ROOT_DIR}"
  require_bayesdag_torch_cuda "train_single" "${CAUSAL_META_BAYESDAG_PYTHON}"
fi

if [[ "${MODEL}" == "dibs" ]]; then
  require_dibs_jax_gpu "train_single" "${MAIN_PYTHON}"
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
export HYDRA_FULL_ERROR=1
export PYTHONFAULTHANDLER=1

"${CAUSAL_META_BIN}" \
  --config-name "${CONFIG_NAME}" \
  "model=${MODEL}" \
  "name=${RUN_NAME}" \
  "$@"
