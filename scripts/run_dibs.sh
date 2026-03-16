#!/usr/bin/env bash
#SBATCH --job-name=cm_dibs
#SBATCH --partition=GPU-a100
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=72:00:00
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR_FROM_SCRIPT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "[run_dibs] No SLURM allocation detected; submitting with sbatch." >&2
  exec sbatch --export="ALL,CAUSAL_META_ROOT_DIR=${ROOT_DIR_FROM_SCRIPT}" "$0" "$@"
fi

ROOT_DIR="${CAUSAL_META_ROOT_DIR:-${SLURM_SUBMIT_DIR:-${ROOT_DIR_FROM_SCRIPT}}}"
cd "${ROOT_DIR}"

CONFIG_NAME="${1:-full_multimodel}"
RUN_NAME="${2:-dibs_${SLURM_JOB_ID:-manual}}"
if [[ "$#" -ge 2 ]]; then
  shift 2
else
  shift "$#"
fi

mkdir -p "${ROOT_DIR}/experiments/runs/${RUN_NAME}"

VENV_DIR="${VENV_DIR:-${UV_PROJECT_ENVIRONMENT:-.venv}}"
if [[ "${VENV_DIR}" != /* ]]; then
  VENV_DIR="${ROOT_DIR}/${VENV_DIR}"
fi

MAIN_PYTHON="${VENV_DIR}/bin/python"
if [[ ! -x "${MAIN_PYTHON}" ]]; then
  MAIN_PYTHON="python3"
fi

if [[ -n "${SLURM_JOB_NODELIST:-}" ]]; then
  MASTER_ADDR="${MASTER_ADDR:-$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)}"
else
  MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
fi
export MASTER_ADDR

job_id_for_port="${SLURM_JOB_ID:-0}"
export MASTER_PORT
MASTER_PORT="${MASTER_PORT:-$((10000 + (job_id_for_port % 50000)))}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export HYDRA_FULL_ERROR=1
export PYTHONFAULTHANDLER=1

# Prevent login-node GPU visibility leakage
unset CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL 2>/dev/null || true

echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "SLURM_NTASKS=${SLURM_NTASKS:-unset}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
echo "SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "GPU_DEVICE_ORDINAL=${GPU_DEVICE_ORDINAL:-unset}"
echo "Launching ${SLURM_NTASKS:-1} tasks via srun"

echo "Per-task GPU mapping (preflight):"
srun bash -lc 'uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | head -n 1); echo "task=${SLURM_PROCID:-unset} local=${SLURM_LOCALID:-unset} CVD=${CUDA_VISIBLE_DEVICES:-unset} GDO=${GPU_DEVICE_ORDINAL:-unset} UUID=${uuid:-unset}"'

srun "${MAIN_PYTHON}" -m causal_meta.main \
  --config-name "${CONFIG_NAME}" \
  "model=dibs" \
  "name=${RUN_NAME}" \
  "$@"