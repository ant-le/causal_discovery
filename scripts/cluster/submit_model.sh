#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <model> [config_name] [run_name] [hydra_overrides...]" >&2
  exit 1
fi

MODEL="$1"
CONFIG_NAME="${2:-full_multimodel}"
RUN_NAME="${3:-rq1_${MODEL}_only}"

if [[ "$#" -ge 3 ]]; then
  shift 3
else
  shift "$#"
fi

PARTITION="${PARTITION:-GPU-a100s}"
GPU_TYPE="${GPU_TYPE:-}"
TIME_HOURS="${TIME_HOURS:-72}"
CPUS_PER_GPU="${CPUS_PER_GPU:-5}"
GPU_REQUEST_MODE="${GPU_REQUEST_MODE:-auto}"
SET_GPUS_PER_TASK="${SET_GPUS_PER_TASK:-0}"

case "${MODEL}" in
  avici|bcnp)
    JOB_SCRIPT="${ROOT_DIR}/scripts/cluster/train_ddp.sbatch"
    GPUS="${DDP_GPUS:-4}"
    MEM_GB="${DDP_MEM_GB:-200}"
    CPUS_PER_TASK=$((GPUS * CPUS_PER_GPU))
    if [[ "${SET_GPUS_PER_TASK}" == "1" ]]; then
      GPU_TASK_ARGS=(--gpus-per-task="${GPUS}")
    else
      GPU_TASK_ARGS=()
    fi
    ;;
  *)
    JOB_SCRIPT="${ROOT_DIR}/scripts/cluster/train_single.sbatch"
    GPUS="${SINGLE_GPUS:-1}"
    MEM_GB="${SINGLE_MEM_GB:-50}"
    CPUS_PER_TASK="${CPUS_PER_GPU}"
    GPU_TASK_ARGS=()
    ;;
esac

RUN_DIR="${ROOT_DIR}/experiments/runs/${RUN_NAME}"
mkdir -p "${RUN_DIR}"

LOG_OUT="${RUN_DIR}/slurm_%j.out"
LOG_ERR="${RUN_DIR}/slurm_%j.err"

SBATCH_ARGS=(
  --partition="${PARTITION}"
  --chdir="${ROOT_DIR}"
  --nodes=1
  --ntasks=1
  --cpus-per-task="${CPUS_PER_TASK}"
  --mem="${MEM_GB}G"
  --time="${TIME_HOURS}:00:00"
  --job-name="cm_${MODEL}"
  --output="${LOG_OUT}"
  --error="${LOG_ERR}"
  --export="ALL,NPROC_PER_NODE=${GPUS}"
)

SBATCH_ARGS+=("${GPU_TASK_ARGS[@]}")

gpu_count_spec="${GPUS}"
if [[ -n "${GPU_TYPE}" ]]; then
  gpu_count_spec="${GPU_TYPE}:${GPUS}"
fi

build_mode_args() {
  local mode="$1"
  case "${mode}" in
    gpus-per-node)
      printf -- '--gpus-per-node=%s\n' "${gpu_count_spec}"
      ;;
    gpus)
      printf -- '--gpus=%s\n' "${gpu_count_spec}"
      ;;
    gres)
      if [[ -n "${GPU_TYPE}" ]]; then
        printf -- '--gres=gpu:%s:%s\n' "${GPU_TYPE}" "${GPUS}"
      else
        printf -- '--gres=gpu:%s\n' "${GPUS}"
      fi
      ;;
    *)
      return 1
      ;;
  esac
}

if [[ "${GPU_REQUEST_MODE}" == "auto" ]]; then
  modes=(gpus-per-node gpus gres)
else
  modes=("${GPU_REQUEST_MODE}")
fi

for mode in "${modes[@]}"; do
  mapfile -t mode_args < <(build_mode_args "${mode}") || {
    echo "Unsupported GPU_REQUEST_MODE='${mode}'. Use one of: auto, gpus-per-node, gpus, gres." >&2
    exit 1
  }

  echo "[submit_model] trying GPU request mode='${mode}'" >&2
  if output=$(sbatch "${SBATCH_ARGS[@]}" "${mode_args[@]}" "${JOB_SCRIPT}" "${MODEL}" "${CONFIG_NAME}" "${RUN_NAME}" "$@" 2>&1); then
    printf '%s\n' "${output}"
    exit 0
  fi
  rc=$?

  if [[ "${GPU_REQUEST_MODE}" != "auto" ]]; then
    printf '%s\n' "${output}" >&2
    exit "${rc}"
  fi

  case "${output}" in
    *"Invalid GRES specification"*|*"Invalid generic resource (gres) specification"*|*"Invalid generic resource specification"*|*"unrecognized option '--gpus-per-node'"*|*"unrecognized option '--gpus'"*|*"Requested GRES option unsupported"*)
      echo "[submit_model] mode='${mode}' rejected by Slurm; trying next mode." >&2
      ;;
    *)
      printf '%s\n' "${output}" >&2
      exit "${rc}"
      ;;
  esac
done

echo "[submit_model] all GPU request modes failed."
echo "[submit_model] Try: sinfo -o '%P %G' and scontrol show config | grep -E 'SelectType|GresTypes'" >&2
exit 2
