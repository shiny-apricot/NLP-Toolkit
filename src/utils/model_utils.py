"""Model management utilities."""

from typing import Dict, Any, Tuple
from src.utils.project_logger import ProjectLogger
from src.utils.gpu_utils import GPUManager, GPUConfig
from src.config_loader import ConfigLoader

def load_model_parameters(
    instance_type: str,
    logger: ProjectLogger
) -> Tuple[Dict[str, Any], GPUManager]:
    """Load model settings based on the AWS instance type."""
    config_loader = ConfigLoader("configs/aws_configs")
    config = config_loader.load_instance_config(instance_type)
    
    gpu_config = GPUConfig(
        min_memory_mb=config.memory_gb * 1024,
        max_memory_percent=0.9,
        strategy="data_parallel",
        prefer_bfloat16=config.fp16
    )
    gpu_manager = GPUManager(gpu_config, logger)
    
    max_length = config.max_length
    min_length = max(50, max_length // 4)
    
    params = {
        "model_name": config.model_name,
        "max_length": max_length,
        "min_length": min_length,
        "num_beams": 4,
        "length_penalty": 1.0,
        "early_stopping": True,
        "batch_size": 8,
        "truncation_strategy": "longest_first",
        "stride": 128,
        "show_progress": True,
        "dtype": config.dtype
    }
    
    return params, gpu_manager
