import logging
import os
import platform
import sys

import torch

log = logging.getLogger(__name__)


def log_environment_info() -> None:
    """Logs lightweight environment information for debugging/reproducibility."""
    info = [
        f"Python: {sys.version.split()[0]} ({platform.system()} {platform.release()})",
        f"PyTorch: {torch.__version__}",
        f"CUDA Available: {torch.cuda.is_available()}",
    ]
    
    if torch.cuda.is_available():
        info.append(f"CUDA Device Count: {torch.cuda.device_count()}")
        current = torch.cuda.current_device()
        info.append(f"Current Device: {torch.cuda.get_device_name(current)}")
    
    # Optional: Check for Slurm
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    if slurm_job_id:
        info.append(f"SLURM_JOB_ID: {slurm_job_id}")

    log.info("Environment Report:\n" + "\n".join([f"  - {line}" for line in info]))

