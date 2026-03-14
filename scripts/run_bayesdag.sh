#!/usr/bin/env bash
set -euo pipefail

MODEL="bayesdag"
GPU_TYPE="a100"
GPU_COUNT=1
CPUS_PER_TASK=5
MEM_GB=80
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
    --ntasks=1 \
    --ntasks-per-node=1 \
    --cpus-per-task="${CPUS_PER_TASK}" \
    --gres="gpu:${GPU_TYPE}:${GPU_COUNT}" \
    --mem="${MEM_GB}G" \
    --time="${TIME_LIMIT}" \
    --job-name="cm_${MODEL}" \
    --output="${run_dir}/slurm_%j.out" \
    --error="${run_dir}/slurm_%j.err" \
    --chdir="${ROOT_DIR}" \
    --export="ALL,CAUSAL_META_ROOT_DIR=${ROOT_DIR}" \
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
  if [[ -z "${CAUSAL_META_BAYESDAG_PYTHON:-}" ]]; then
    export CAUSAL_META_BAYESDAG_PYTHON="${ROOT_DIR}/.venv-bayesdag/bin/python"
  fi

  "${main_python}" -m causal_meta.main \
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
