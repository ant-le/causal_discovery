#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LAUNCH_MODE="srun"
exec "${SCRIPT_DIR}/run_avici.sh" "$@"
