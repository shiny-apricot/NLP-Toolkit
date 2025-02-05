from dataclasses import dataclass
import torch
from typing import Optional

@dataclass
class GPUConfig:
    """Configuration for GPU resource management."""
    min_memory_mb: int
    max_memory_percent: float
    strategy: str
    prefer_bfloat16: bool = True
    device_id: Optional[int] = None

class GPUManager:
    """Context manager for GPU memory management."""
    
    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if torch.cuda.is_available() and self.config.device_id is not None:
                torch.cuda.set_device(self.config.device_id)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __init__(self, config: GPUConfig):
        self.config = config
