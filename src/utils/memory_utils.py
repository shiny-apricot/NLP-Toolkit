"""Memory management utilities for large-scale processing."""

import gc
import psutil
from dataclasses import dataclass
import torch
from typing import Optional
from pathlib import Path
import json
import time
from src.utils.project_logger import ProjectLogger

@dataclass
class MemoryStats:
    """Statistics about memory usage."""
    system_used_gb: float
    system_available_gb: float
    system_percent: float
    python_used_gb: float
    timestamp: float

def get_memory_stats() -> MemoryStats:
    """Get current memory statistics."""
    system = psutil.virtual_memory()
    process = psutil.Process()
    
    return MemoryStats(
        system_used_gb=system.used / (1024**3),
        system_available_gb=system.available / (1024**3),
        system_percent=system.percent,
        python_used_gb=process.memory_info().rss / (1024**3),
        timestamp=time.time()
    )

def log_memory_usage(logger: ProjectLogger) -> None:
    """Log current memory usage statistics."""
    stats = get_memory_stats()
    logger.info(
        "Memory usage",
        system_used_gb=stats.system_used_gb,
        system_available_gb=stats.system_available_gb,
        system_percent=stats.system_percent,
        python_used_gb=stats.python_used_gb
    )

def optimize_memory(
    *,
    logger: ProjectLogger,
    gc_collect: bool = True,
    clear_cache: bool = True
) -> None:
    """Optimize memory usage."""
    if gc_collect:
        gc.collect()
    if clear_cache:
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    log_memory_usage(logger)

class MemoryTracker:
    """Context manager for tracking memory usage."""
    
    def __init__(
        self,
        logger: ProjectLogger,
        label: str,
        log_path: Optional[Path] = None
    ):
        self.logger = logger
        self.label = label
        self.log_path = log_path
        self.start_stats = None
        self.end_stats = None

    def __enter__(self):
        self.start_stats = get_memory_stats()
        self.logger.info(f"Starting {self.label}", memory=self.start_stats.__dict__)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_stats = get_memory_stats()
        memory_change = self.end_stats.python_used_gb - self.start_stats.python_used_gb
        
        self.logger.info(
            f"Completed {self.label}",
            memory_change_gb=memory_change,
            end_memory=self.end_stats.__dict__
        )
        
        if self.log_path:
            with open(self.log_path, 'a') as f:
                json.dump({
                    'label': self.label,
                    'start': self.start_stats.__dict__,
                    'end': self.end_stats.__dict__,
                    'memory_change_gb': memory_change
                }, f)
                f.write('\n')
