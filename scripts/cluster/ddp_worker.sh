#!/bin/bash
set -euo pipefail

MAIN_PYTHON="${1:?Usage: ddp_worker.sh <main_python> <args...>}"
shift

export RANK="${SLURM_PROCID}"
export LOCAL_RANK="${SLURM_LOCALID}"
export WORLD_SIZE="${SLURM_NTASKS}"

exec "${MAIN_PYTHON}" -m causal_meta.main "$@"
