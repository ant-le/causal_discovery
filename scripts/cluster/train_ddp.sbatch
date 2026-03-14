#!/bin/bash
set -euo pipefail

ROOT_DIR="${CAUSAL_META_ROOT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${ROOT_DIR}"
VENV_DIR="${VENV_DIR:-${UV_PROJECT_ENVIRONMENT:-.venv}}"
if [[ "${VENV_DIR}" != /* ]]; then
  VENV_DIR="${ROOT_DIR}/${VENV_DIR}"
fi
MAIN_PYTHON="${VENV_DIR}/bin/python"
DDP_WORKER="${ROOT_DIR}/scripts/cluster/ddp_worker.sh"

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

readarray -t hosts < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
MASTER_ADDR="${MASTER_ADDR:-${hosts[0]}}"
MASTER_PORT="${MASTER_PORT:-$((10000 + SLURM_JOB_ID % 50000))}"

srun \
  --ntasks="${NPROC_PER_NODE}" \
  --ntasks-per-node="${NPROC_PER_NODE}" \
  --cpus-per-task="${SLURM_CPUS_PER_TASK:-1}" \
  --export=ALL,MASTER_ADDR="${MASTER_ADDR}",MASTER_PORT="${MASTER_PORT}" \
  "${DDP_WORKER}" \
  "${MAIN_PYTHON}" \
  --config-name "${CONFIG_NAME}" \
  "model=${MODEL}" \
  "name=${RUN_NAME}" \
  "$@"
