#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"${SCRIPT_DIR}/run_bcnp.sh" smoke_multimodel bcnp_multigpu_smoke \
  trainer.max_steps=2 \
  trainer.val_check_interval=2 \
  trainer.log_every_n_steps=1 \
  inference.n_samples=2
