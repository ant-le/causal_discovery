#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Auto-load BayesDAG external python from bootstrap snippet if available.
if [[ -z "${CAUSAL_META_BAYESDAG_PYTHON:-}" ]] && [[ -f "${ROOT_DIR}/.bootstrap_env.sh" ]]; then
  # shellcheck disable=SC1090
  source "${ROOT_DIR}/.bootstrap_env.sh"
fi

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
GPU_TYPE="${GPU_TYPE:-a100s}"
TIME_HOURS="${TIME_HOURS:-72}"
CPUS_PER_GPU="${CPUS_PER_GPU:-5}"
GPU_REQUEST_MODE="${GPU_REQUEST_MODE:-gres}"
SET_GPUS_PER_TASK="${SET_GPUS_PER_TASK:-1}"

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

case "${GPU_REQUEST_MODE}" in
  gpus-per-node)
    SBATCH_ARGS+=(--gpus-per-node="${GPUS}")
    ;;
  gpus)
    SBATCH_ARGS+=(--gpus="${GPUS}")
    ;;
  gres)
    if [[ -n "${GPU_TYPE}" ]]; then
      SBATCH_ARGS+=(--gres="gpu:${GPU_TYPE}:${GPUS}")
    else
      SBATCH_ARGS+=(--gres="gpu:${GPUS}")
    fi
    ;;
  *)
    echo "Unsupported GPU_REQUEST_MODE='${GPU_REQUEST_MODE}'. Use one of: gpus-per-node, gpus, gres." >&2
    exit 1
    ;;
esac

SBATCH_ARGS+=("${GPU_TASK_ARGS[@]}")

sbatch "${SBATCH_ARGS[@]}" "${JOB_SCRIPT}" "${MODEL}" "${CONFIG_NAME}" "${RUN_NAME}" "$@"
