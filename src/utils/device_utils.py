"""Device management utilities for deep learning models."""

import torch
from src.utils.project_logger import ProjectLogger
from src.utils.gpu_utils import GPUManager

def setup_compute_device(
    device_preference: str,
    gpu_manager: GPUManager,
    logger: ProjectLogger
) -> torch.device:
    """Choose the best available device (GPU or CPU) for processing."""
    try:
        if device_preference != "auto":
            return torch.device(device_preference)
        
        device = gpu_manager.select_device()
        logger.info("Device selected", device=str(device))
        return device
        
    except Exception as e:
        logger.warning(
            "Device selection failed, falling back to CPU",
            error=str(e)
        )
        return torch.device("cpu")

def cleanup_device_resources(model: torch.nn.Module) -> None:
    """Free up GPU memory and resources."""
    if hasattr(model, 'cpu'):
        model.cpu()
    torch.cuda.empty_cache()
