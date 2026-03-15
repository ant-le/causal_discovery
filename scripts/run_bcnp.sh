#!/usr/bin/env bash
set -euo pipefail

MODEL="bcnp"
GPU_TYPE="a100"
GPU_COUNT=4
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

  local -a common_args=(
    --partition="${PARTITION}"
    --nodes=1
    --tasks-per-node=1
    --cpus-per-task="${CPUS_PER_TASK}"
    --mem="${MEM_GB}G"
    --time="${TIME_LIMIT}"
    --job-name="cm_${MODEL}"
    --output="${run_dir}/slurm_%j.out"
    --error="${run_dir}/slurm_%j.err"
    --chdir="${ROOT_DIR}"
    --export="ALL,CAUSAL_META_ROOT_DIR=${ROOT_DIR},NPROC_PER_NODE=${GPU_COUNT}"
  )

  local output
  local -a gpu_modes=(
    "--gpus-per-task=${GPU_COUNT}"
    "--gpus-per-node=${GPU_COUNT}"
    "--gres=gpu:${GPU_TYPE}:${GPU_COUNT}"
    "--gres=gpu:${GPU_COUNT}"
  )

  local mode
  for mode in "${gpu_modes[@]}"; do
    if output=$(env -u CUDA_VISIBLE_DEVICES sbatch "${common_args[@]}" "${mode}" "$0" "${config_name}" "${run_name}" "$@" 2>&1); then
      printf '%s\n' "${output}"
      return 0
    fi

    if [[ "${output}" == *"Invalid generic resource"* || "${output}" == *"Invalid GRES"* || "${output}" == *"unrecognized option '--gpus-per-node'"* ]]; then
      echo "[run_${MODEL}] GPU request mode rejected (${mode}), trying fallback..." >&2
      continue
    fi

    printf '%s\n' "${output}" >&2
    exit 1
  done

  printf '%s\n' "${output}" >&2
  exit 1
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
  local torchrun_bin="${venv_dir}/bin/torchrun"

  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
  export HYDRA_FULL_ERROR=1
  export PYTHONFAULTHANDLER=1

  # Clear GPU visibility vars that may have leaked from the login node via
  # --export=ALL.  Unsetting lets SLURM's cgroup/prolog GPU binding take
  # effect based on the actual gres allocation.
  unset CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL 2>/dev/null || true

  # Diagnostic: log SLURM GPU allocation details
  echo "[run_${MODEL}] SLURM_JOB_ID=${SLURM_JOB_ID:-unset}" >&2
  echo "[run_${MODEL}] SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-unset}" >&2
  echo "[run_${MODEL}] SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}" >&2
  echo "[run_${MODEL}] SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-unset}" >&2
  echo "[run_${MODEL}] GPU_DEVICE_ORDINAL=${GPU_DEVICE_ORDINAL:-unset}" >&2
  echo "[run_${MODEL}] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}" >&2
  echo "[run_${MODEL}] NPROC_PER_NODE=${NPROC_PER_NODE:-unset}" >&2
  nvidia-smi -L 2>/dev/null || echo "[run_${MODEL}] nvidia-smi not available" >&2
  echo "[run_${MODEL}] MIG mode:" >&2
  nvidia-smi --query-gpu=mig.mode.current --format=csv 2>/dev/null || echo "  query failed" >&2
  echo "[run_${MODEL}] Per-GPU memory (full A100-80GB = ~81920 MiB):" >&2
  nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null || echo "  query failed" >&2

  # Detect visible GPUs via nvidia-smi instead of Python/torch.
  # Spawning Python here would call cudaInit(), which on some SLURM
  # cgroup configurations can lock or partition the GPU assignment
  # before torchrun gets a chance to spawn its workers.
  local visible_gpu_count
  visible_gpu_count="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')"
  if [[ "${visible_gpu_count}" -lt 1 ]]; then
    echo "[run_${MODEL}] ERROR: no CUDA GPUs visible in job allocation (nvidia-smi returned 0)." >&2
    exit 2
  fi
  echo "[run_${MODEL}] nvidia-smi reports ${visible_gpu_count} GPU(s) visible." >&2

  local nproc_per_node="${NPROC_PER_NODE:-${GPU_COUNT}}"
  if [[ "${visible_gpu_count}" -lt "${nproc_per_node}" ]]; then
    echo "[run_${MODEL}] WARNING: requested ${nproc_per_node} GPUs, but ${visible_gpu_count} are visible. Adjusting nproc_per_node." >&2
    nproc_per_node="${visible_gpu_count}"
  fi

  "${torchrun_bin}" --standalone --nproc_per_node "${nproc_per_node}" -m causal_meta.main \
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
