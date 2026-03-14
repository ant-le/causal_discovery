#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
VENV_DIR="${VENV_DIR:-${UV_PROJECT_ENVIRONMENT:-.venv}}"
if [[ "${VENV_DIR}" != /* ]]; then
  VENV_DIR="${ROOT_DIR}/${VENV_DIR}"
fi
MAIN_PYTHON="${VENV_DIR}/bin/python"
TORCHRUN_BIN="${VENV_DIR}/bin/torchrun"

MODEL="${1:?Usage: train_ddp.sbatch <model> [config_name] [run_name] [hydra_overrides...] }"
CONFIG_NAME="${2:-full_multimodel}"
RUN_NAME="${3:-rq1_${MODEL}_ddp}"

if [[ "$#" -ge 3 ]]; then
  shift 3
else
  shift "$#"
fi

NPROC_PER_NODE="${NPROC_PER_NODE:-4}"

if [[ -z "${OMP_NUM_THREADS:-}" ]]; then
  cpus_per_task="${SLURM_CPUS_PER_TASK:-1}"
  omp_threads=$((cpus_per_task / NPROC_PER_NODE))
  if [[ "${omp_threads}" -lt 1 ]]; then
    omp_threads=1
  fi
  export OMP_NUM_THREADS="${omp_threads}"
fi
export HYDRA_FULL_ERROR=1
export PYTHONFAULTHANDLER=1

VISIBLE_GPU_COUNT="$("${MAIN_PYTHON}" -c 'import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)')"
echo "[train_ddp] requested_nproc=${NPROC_PER_NODE} visible_gpus=${VISIBLE_GPU_COUNT}" >&2
if [[ "${VISIBLE_GPU_COUNT}" -lt "${NPROC_PER_NODE}" ]]; then
  echo "[train_ddp] ERROR: fewer visible GPUs than requested processes." >&2
  echo "[train_ddp] Hint: ensure the Slurm request exposes ${NPROC_PER_NODE} GPUs to this task." >&2
  echo "[train_ddp] Try SET_GPUS_PER_TASK=1 or adjust GPU_REQUEST_MODE in submit_model.sh." >&2
  exit 2
fi

"${TORCHRUN_BIN}" --standalone --nproc_per_node "${NPROC_PER_NODE}" -m causal_meta.main \
  --config-name "${CONFIG_NAME}" \
  "model=${MODEL}" \
  "name=${RUN_NAME}" \
  "$@"
