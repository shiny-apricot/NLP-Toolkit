import torch
import torch.cuda as cuda
from typing import Optional, List, Dict, Union, Tuple
from dataclasses import dataclass
import logging
import psutil
import gc
from contextlib import contextmanager

@dataclass
class GPUConfig:
    """Configuration for GPU resource management."""
    enable_memory_optimization: bool = True
    enable_auto_device_selection: bool = True
    min_memory_required: int = 4 * 1024  # 4GB in MB
    max_memory_usage: float = 0.9  # 90% of available memory
    prefer_bfloat16: bool = True
    multi_gpu_strategy: str = "data_parallel"  # or "model_parallel"

class GPUManager:
    """Manage GPU resources and optimization."""
    
    def __init__(
        self,
        config: GPUConfig,
        logger: Optional[logging.Logger] = None
    ):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.initialized_devices: List[int] = []
        
    def get_available_devices(self) -> List[int]:
        """Get list of available GPU devices with sufficient memory."""
        available_devices = []
        
        if not torch.cuda.is_available():
            return available_devices
            
        for device_id in range(torch.cuda.device_count()):
            if self._check_device_memory(device_id):
                available_devices.append(device_id)
                
        return available_devices

    def _check_device_memory(self, device_id: int) -> bool:
        """Check if device has sufficient free memory."""
        try:
            total_memory = torch.cuda.get_device_properties(device_id).total_memory
            free_memory = total_memory - torch.cuda.memory_allocated(device_id)
            free_memory_mb = free_memory / 1024 / 1024
            
            return free_memory_mb >= self.config.min_memory_required
        except Exception as e:
            self.logger.warning(f"Failed to check memory for device {device_id}: {str(e)}")
            return False

    def select_device(self) -> torch.device:
        """Select best available device based on memory and computation capability."""
        if not torch.cuda.is_available() or not self.config.enable_auto_device_selection:
            return torch.device("cpu")
            
        available_devices = self.get_available_devices()
        if not available_devices:
            self.logger.warning("No GPU with sufficient memory available")
            return torch.device("cpu")
            
        # Select device with most free memory
        device_memory_free = [
            torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)
            for i in available_devices
        ]
        selected_device = available_devices[device_memory_free.index(max(device_memory_free))]
        
        return torch.device(f"cuda:{selected_device}")

    def get_optimal_batch_size(
        self,
        initial_batch_size: int,
        model_size_mb: float,
        device: Optional[torch.device] = None
    ) -> int:
        """Calculate optimal batch size based on available memory."""
        if device is None:
            device = self.select_device()
            
        if device.type == "cpu":
            # Use system memory for CPU
            available_memory = psutil.virtual_memory().available / 1024 / 1024
        else:
            total_memory = torch.cuda.get_device_properties(device).total_memory / 1024 / 1024
            allocated_memory = torch.cuda.memory_allocated(device) / 1024 / 1024
            available_memory = total_memory - allocated_memory
            
        # Calculate maximum possible batch size
        max_batch_size = int((available_memory * self.config.max_memory_usage) / model_size_mb)
        optimal_batch_size = min(initial_batch_size, max_batch_size)
        
        return max(1, optimal_batch_size)  # Ensure at least batch size of 1

    def optimize_model_memory(
        self,
        model: torch.nn.Module,
        prefer_bfloat16: Optional[bool] = None
    ) -> torch.nn.Module:
        """Optimize model memory usage."""
        if not self.config.enable_memory_optimization:
            return model
            
        use_bfloat16 = prefer_bfloat16 if prefer_bfloat16 is not None else self.config.prefer_bfloat16
        
        if use_bfloat16 and torch.cuda.is_available():
            model = model.to(torch.bfloat16)
            
        return model

    def setup_multi_gpu(
        self,
        model: torch.nn.Module,
        strategy: Optional[str] = None
    ) -> torch.nn.Module:
        """Setup model for multi-GPU training."""
        if not torch.cuda.is_available() or torch.cuda.device_count() <= 1:
            return model
            
        strategy = strategy or self.config.multi_gpu_strategy
        
        if strategy == "data_parallel":
            return torch.nn.DataParallel(model)
        elif strategy == "model_parallel":
            # Implement model parallelism if needed
            self.logger.warning("Model parallelism not implemented yet")
            return model
        else:
            raise ValueError(f"Unknown multi-GPU strategy: {strategy}")

    @contextmanager
    def track_memory(self, device: Optional[torch.device] = None):
        """Context manager to track memory usage."""
        if device is None:
            device = self.select_device()
            
        try:
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                initial_memory = torch.cuda.memory_allocated(device)
            
            yield
            
        finally:
            if device.type == "cuda":
                final_memory = torch.cuda.memory_allocated(device)
                peak_memory = torch.cuda.max_memory_allocated(device)
                self.logger.info(
                    "Memory usage",
                    initial_mb=initial_memory / 1024 / 1024,
                    final_mb=final_memory / 1024 / 1024,
                    peak_mb=peak_memory / 1024 / 1024
                )

    def cleanup(self):
        """Clean up GPU memory."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            for device in self.initialized_devices:
                try:
                    torch.cuda.reset_peak_memory_stats(device)
                except:
                    pass

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()

# Example usage
if __name__ == "__main__":
    config = GPUConfig(
        enable_memory_optimization=True,
        prefer_bfloat16=True
    )
    
    gpu_manager = GPUManager(config)
    
    # Use as context manager
    with gpu_manager:
        device = gpu_manager.select_device()
        print(f"Selected device: {device}")
        
        # Track memory usage
        with gpu_manager.track_memory(device):
            # Your GPU operations here
            pass
