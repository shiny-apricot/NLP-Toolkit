from dataclasses import dataclass
from typing import Optional, List, Literal, AsyncContextManager
import torch
import asyncio
import logging
from contextlib import asynccontextmanager
import psutil
import gc

class GPUError(Exception):
    """Base exception for GPU-related errors."""
    pass

class InsufficientMemoryError(GPUError):
    """Raised when there isn't enough GPU memory."""
    pass

class DeviceNotFoundError(GPUError):
    """Raised when no suitable device is found."""
    pass

class MemoryStats:
    """Immutable memory statistics."""
    def __init__(self, allocated_mb: float, reserved_mb: float, free_mb: float, total_mb: float, peak_mb: float) -> None:
        self.allocated_mb = allocated_mb
        self.reserved_mb = reserved_mb
        self.free_mb = free_mb
        self.total_mb = total_mb
        self.peak_mb = peak_mb

class GPUManager:
    def __init__(
        self,
        *,
        min_memory_mb: int = 4 * 1024,  # 4GB
        max_memory_percent: float = 0.9,
        multi_gpu_strategy: Literal["data_parallel", "model_parallel"] = "data_parallel",
        prefer_bfloat16: bool = True,
        logger: Optional[logging.Logger] = None
    ) -> None:
        self.min_memory_mb = min_memory_mb
        self.max_memory_percent = max_memory_percent
        self.multi_gpu_strategy = multi_gpu_strategy
        self.prefer_bfloat16 = prefer_bfloat16
        self.logger = logger or logging.getLogger(__name__)
        self._active_devices: List[int] = []

    def get_memory_stats(self, device_id: int) -> MemoryStats:
        """Get detailed memory statistics for a device."""
        if not torch.cuda.is_available():
            raise DeviceNotFoundError("CUDA not available")

        props = torch.cuda.get_device_properties(device_id)
        total = props.total_memory / 1024 / 1024
        allocated = torch.cuda.memory_allocated(device_id) / 1024 / 1024
        reserved = torch.cuda.memory_reserved(device_id) / 1024 / 1024
        free = total - allocated
        peak = torch.cuda.max_memory_allocated(device_id) / 1024 / 1024

        return MemoryStats(
            allocated_mb=allocated,
            reserved_mb=reserved,
            free_mb=free,
            total_mb=total,
            peak_mb=peak
        )

    def select_device(self) -> torch.device:
        """Select best available device based on memory and compute capability."""
        if not torch.cuda.is_available():
            return torch.device("cpu")

        best_device = -1
        max_free_memory = 0

        for device_id in range(torch.cuda.device_count()):
            try:
                stats = self.get_memory_stats(device_id)
                if (
                    stats.free_mb >= self.min_memory_mb
                    and stats.free_mb > max_free_memory
                ):
                    best_device = device_id
                    max_free_memory = stats.free_mb
            except Exception as e:
                self.logger.warning(f"Error checking device {device_id}: {e}")

        if best_device == -1:
            return torch.device("cpu")

        return torch.device(f"cuda:{best_device}")

    def optimize_model(
        self,
        model: torch.nn.Module,
        device: Optional[torch.device] = None,
        *,
        use_bfloat16: Optional[bool] = None
    ) -> torch.nn.Module:
        """Optimize model for inference or training."""
        device = device or self.select_device()
        model = model.to(device)

        use_bf16 = self.prefer_bfloat16 if use_bfloat16 is None else use_bfloat16
        if device.type == "cuda" and use_bf16:
            model = model.to(torch.bfloat16)

        if (
            device.type == "cuda"
            and torch.cuda.device_count() > 1
            and self.multi_gpu_strategy == "data_parallel"
        ):
            model = torch.nn.DataParallel(model)

        return model

    def get_optimal_batch_size(
        self,
        model_size_mb: float,
        initial_batch_size: int,
        device: Optional[torch.device] = None
    ) -> int:
        """Calculate optimal batch size based on available memory."""
        device = device or self.select_device()

        if device.type == "cpu":
            available_memory = psutil.virtual_memory().available / 1024 / 1024
        else:
            stats = self.get_memory_stats(device.index)
            available_memory = stats.free_mb

        max_batch_size = int((available_memory * self.max_memory_percent) / model_size_mb)
        return max(1, min(initial_batch_size, max_batch_size))

    @asynccontextmanager
    async def memory_tracker(self, device: torch.device) -> AsyncContextManager[None]:
        """Async context manager for tracking memory usage."""
        if device.type != "cuda":
            yield
            return

        initial_stats = self.get_memory_stats(device.index)
        try:
            yield
        finally:
            final_stats = self.get_memory_stats(device.index)
            self.logger.info(
                "Memory usage",
                extra={
                    "initial_mb": initial_stats.allocated_mb,
                    "final_mb": final_stats.allocated_mb,
                    "peak_mb": final_stats.peak_mb,
                    "device": device.index
                }
            )

    async def cleanup(self) -> None:
        """Asynchronously clean up GPU memory."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            await asyncio.gather(*[
                self._reset_device(device)
                for device in self._active_devices
            ])

    async def _reset_device(self, device_id: int) -> None:
        """Reset a single device's memory stats."""
        try:
            torch.cuda.reset_peak_memory_stats(device_id)
        except Exception as e:
            self.logger.warning(f"Failed to reset device {device_id}: {e}")

    async def __aenter__(self) -> 'GPUManager':
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.cleanup()
from typing import Optional, List, Literal, AsyncContextManager, cast
import torch
import asyncio
import logging
from contextlib import asynccontextmanager
import psutil
import gc

class GPUError(Exception):
    """Base exception for GPU-related errors."""
    pass

class InsufficientMemoryError(GPUError):
    """Raised when there isn't enough GPU memory."""
    pass

class DeviceNotFoundError(GPUError):
    """Raised when no suitable device is found."""
    pass

@dataclass(frozen=True)
class MemoryStats:
    """Immutable memory statistics."""
    allocated_mb: float
    reserved_mb: float
    free_mb: float
    total_mb: float
    peak_mb: float

@dataclass(frozen=True)
class GPUConfig:
    """Immutable GPU configuration."""
    min_memory_mb: int = 4 * 1024  # 4GB
    max_memory_percent: float = 0.9  # 90%
    strategy: Literal["data_parallel", "model_parallel"] = "data_parallel"
    prefer_bfloat16: bool = True

class GPUManager:
    def __init__(
        self,
        config: GPUConfig,
        logger: Optional[logging.Logger] = None
    ) -> None:
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self._active_devices: List[int] = []

    def get_memory_stats(self, device_id: int) -> MemoryStats:
        """Get detailed memory statistics for a device."""
        if not torch.cuda.is_available():
            raise DeviceNotFoundError("CUDA not available")

        props = torch.cuda.get_device_properties(device_id)
        total = props.total_memory / 1024 / 1024
        allocated = torch.cuda.memory_allocated(device_id) / 1024 / 1024
        reserved = torch.cuda.memory_reserved(device_id) / 1024 / 1024
        free = total - allocated
        peak = torch.cuda.max_memory_allocated(device_id) / 1024 / 1024

        return MemoryStats(
            allocated_mb=allocated,
            reserved_mb=reserved,
            free_mb=free,
            total_mb=total,
            peak_mb=peak
        )

    def select_device(self) -> torch.device:
        """Select best available device based on memory and compute capability."""
        if not torch.cuda.is_available():
            return torch.device("cpu")

        best_device = -1
        max_free_memory = 0

        for device_id in range(torch.cuda.device_count()):
            try:
                stats = self.get_memory_stats(device_id)
                if (
                    stats.free_mb >= self.config.min_memory_mb
                    and stats.free_mb > max_free_memory
                ):
                    best_device = device_id
                    max_free_memory = stats.free_mb
            except Exception as e:
                self.logger.warning(f"Error checking device {device_id}: {e}")

        if best_device == -1:
            return torch.device("cpu")

        return torch.device(f"cuda:{best_device}")

    def warmup_device(self, device: torch.device) -> None:
        """Warm up a GPU device to ensure it's ready for computation."""
        if device.type == "cuda":
            # Perform a small computation to warm up the device
            torch.ones(1, device=device).half().float()

    def optimize_model(
        self,
        model: torch.nn.Module,
        device: Optional[torch.device] = None
    ) -> torch.nn.Module:
        """Optimize model for inference or training."""
        device = device or self.select_device()
        model = model.to(device)

        if device.type == "cuda" and self.config.prefer_bfloat16:
            model = model.to(torch.bfloat16)

        if (
            device.type == "cuda"
            and torch.cuda.device_count() > 1
            and self.config.strategy == "data_parallel"
        ):
            model = torch.nn.DataParallel(model)

        return model

    @asynccontextmanager
    async def memory_tracker(self, device: torch.device) -> AsyncContextManager[None]:
        """Async context manager for tracking memory usage."""
        if device.type != "cuda":
            yield
            return

        initial_stats = self.get_memory_stats(device.index)
        try:
            yield
        finally:
            final_stats = self.get_memory_stats(device.index)
            self.logger.info(
                "Memory usage",
                extra={
                    "initial_mb": initial_stats.allocated_mb,
                    "final_mb": final_stats.allocated_mb,
                    "peak_mb": final_stats.peak_mb,
                    "device": device.index
                }
            )

    async def cleanup(self) -> None:
        """Asynchronously clean up GPU memory."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            await asyncio.gather(*[
                self._reset_device(device)
                for device in self._active_devices
            ])

    async def _reset_device(self, device_id: int) -> None:
        """Reset a single device's memory stats."""
        try:
            torch.cuda.reset_peak_memory_stats(device_id)
        except Exception as e:
            self.logger.warning(f"Failed to reset device {device_id}: {e}")

    async def __aenter__(self) -> 'GPUManager':
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.cleanup()

# Example usage
if __name__ == "__main__":
    config = GPUConfig(
        min_memory_mb=4 * 1024,
        max_memory_percent=0.9,
        strategy="data_parallel",
        prefer_bfloat16=True
    )
    
    gpu_manager = GPUManager(config)
    
    # Use as context manager
    async def main():
        async with gpu_manager:
            device = gpu_manager.select_device()
            print(f"Selected device: {device}")
            
            # Track memory usage
            async with gpu_manager.memory_tracker(device):
                # Your GPU operations here
                pass

    asyncio.run(main())
