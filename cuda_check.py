"""
CUDA Check Utility for RIPER-Ω System
Verifies CUDA availability, logs fallbacks, and checks versions/drivers.
"""

import logging
import torch
import subprocess
from typing import Dict

logger = logging.getLogger(__name__)

def check_cuda() -> bool:
    """Check if CUDA is available"""
    available = torch.cuda.is_available()
    if available:
        logger.info("✅ CUDA is available")
        logger.info(f"GPU Device: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA Version: {torch.version.cuda}")
        # Check NVIDIA driver version
        try:
            driver_version = subprocess.check_output("nvidia-smi --query-gpu=driver_version --format=csv,noheader", shell=True).decode().strip()
            logger.info(f"NVIDIA Driver Version: {driver_version}")
        except Exception as e:
            logger.warning(f"Could not retrieve driver version: {e}")
    else:
        logger.warning("⚠️ CUDA not available - falling back to CPU")
        # Log potential issues
        logger.info(f"PyTorch Version: {torch.__version__}")
        logger.info(f"CUDA Compiled: {torch.version.cuda is not None}")
    return available

def assert_cuda() -> None:
    """Assert CUDA availability or raise exception"""
    if not check_cuda():
        raise RuntimeError("CUDA not available - halting execution")

if __name__ == "__main__":
    check_cuda()
