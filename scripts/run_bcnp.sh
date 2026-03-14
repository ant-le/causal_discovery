#!/usr/bin/env bash
set -euo pipefail

MODEL="bcnp"
GPU_TYPE="a100"
GPU_COUNT=5
CPUS_PER_TASK=5
MEM_GB=250
PARTITION="GPU-a100"
TIME_LIMIT="72:00:00"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR_DEFAULT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT_DIR="${CAUSAL_META_ROOT_DIR:-${ROOT_DIR_DEFAULT}}"

submit_job() {
  local config_name="$1"
  local run_name="$2"
  shift 2

  local run_dir="${ROOT_DIR}/experiments/runs/${run_name}"
  mkdir -p "${run_dir}"

  env -u CUDA_VISIBLE_DEVICES sbatch \
    --partition="${PARTITION}" \
    --nodes=1 \
    --ntasks="${GPU_COUNT}" \
    --ntasks-per-node="${GPU_COUNT}" \
    --cpus-per-task="${CPUS_PER_TASK}" \
    --gres="gpu:${GPU_TYPE}:${GPU_COUNT}" \
    --mem="${MEM_GB}G" \
    --time="${TIME_LIMIT}" \
    --job-name="cm_${MODEL}" \
    --output="${run_dir}/slurm_%j.out" \
    --error="${run_dir}/slurm_%j.err" \
    --chdir="${ROOT_DIR}" \
    --export="ALL,CAUSAL_META_ROOT_DIR=${ROOT_DIR},NPROC_PER_NODE=${GPU_COUNT}" \
    "$0" "${config_name}" "${run_name}" "$@"
}

run_job() {
  local config_name="$1"
  local run_name="$2"
  shift 2

  cd "${ROOT_DIR}"
  local venv_dir="${VENV_DIR:-${UV_PROJECT_ENVIRONMENT:-.venv}}"
  if [[ "${venv_dir}" != /* ]]; then
    venv_dir="${ROOT_DIR}/${venv_dir}"
  fi
  local main_python="${venv_dir}/bin/python"

  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-${CPUS_PER_TASK}}}"
  export HYDRA_FULL_ERROR=1
  export PYTHONFAULTHANDLER=1

  readarray -t hosts < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
  local master_addr="${MASTER_ADDR:-${hosts[0]}}"
  local master_port="${MASTER_PORT:-$((10000 + SLURM_JOB_ID % 50000))}"

  local worker
  worker="$(mktemp "${TMPDIR:-/tmp}/cm_ddp_worker.XXXXXX")"
  cat > "${worker}" <<'EOF'
#!/bin/bash
set -euo pipefail
export RANK="${SLURM_PROCID}"
export LOCAL_RANK="${SLURM_LOCALID}"
export WORLD_SIZE="${SLURM_NTASKS}"
exec "$@"
EOF
  chmod +x "${worker}"
  trap 'rm -f "${worker}"' EXIT

  srun \
    --ntasks="${GPU_COUNT}" \
    --ntasks-per-node="${GPU_COUNT}" \
    --cpus-per-task="${SLURM_CPUS_PER_TASK:-${CPUS_PER_TASK}}" \
    --export=ALL,MASTER_ADDR="${master_addr}",MASTER_PORT="${master_port}" \
    "${worker}" \
    "${main_python}" \
    -m causal_meta.main \
    --config-name "${config_name}" \
    "model=${MODEL}" \
    "name=${run_name}" \
    "$@"
}

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  config_name="${1:-full_multimodel}"
  run_name="${2:-rq1_${MODEL}_only}"
  if [[ "$#" -ge 2 ]]; then
    shift 2
  else
    shift "$#"
  fi
  submit_job "${config_name}" "${run_name}" "$@"
else
  config_name="${1:-full_multimodel}"
  run_name="${2:-rq1_${MODEL}_only}"
  if [[ "$#" -ge 2 ]]; then
    shift 2
  else
    shift "$#"
  fi
  run_job "${config_name}" "${run_name}" "$@"
fi
