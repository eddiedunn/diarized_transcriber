"""GPU verification and management utilities."""

import logging
import torch
from typing import Literal, Dict
from ..exceptions import GPUConfigError

logger = logging.getLogger(__name__)

def verify_gpu_requirements() -> Literal["cuda"]:
    """
    Verify GPU requirements and provide detailed feedback.
    
    Returns:
        Literal["cuda"]: The string "cuda" if GPU is available
        
    Raises:
        GPUConfigError: If GPU requirements are not met
    """
    if not torch.cuda.is_available():
        details = {
            "cuda_available": torch.cuda.is_available(),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        }
        raise GPUConfigError(
            "CUDA is not available. Please ensure:\n"
            "1. You have an NVIDIA GPU\n"
            "2. CUDA toolkit is installed\n"
            "3. PyTorch with CUDA support is installed\n"
            f"Current configuration: {details}"
        )
    
    return "cuda"

def get_gpu_info() -> Dict[str, str]:
    """
    Get detailed information about the available GPU.
    
    Returns:
        Dict[str, str]: Dictionary containing GPU information
    """
    info = {
        "gpu_name": torch.cuda.get_device_name(0),
        "cuda_version": torch.version.cuda,
        "gpu_memory_gb": f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}"
    }
    
    logger.info("GPU detected: %(gpu_name)s", info)
    logger.info("CUDA version: %(cuda_version)s", info)
    logger.info("Available GPU memory: %(gpu_memory_gb)s GB", info)
    
    return info

def cleanup_gpu_memory() -> None:
    """
    Clean up GPU memory by clearing cache and garbage collecting.
    """
    import gc
    torch.cuda.empty_cache()
    gc.collect()
