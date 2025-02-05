from dataclasses import dataclass
from typing import Optional, Dict, Any
from pathlib import Path
import json
import logging
import time

class TrainingCallback:
    """Base class for training callbacks."""
    
    def on_training_start(self, trainer: Any):
        """Called when training starts."""
        pass

    def on_training_end(self, trainer: Any, training_time: float):
        """Called when training ends."""
        pass

    def on_epoch_start(self, trainer: Any, epoch: int):
        """Called at the start of each epoch."""
        pass

    def on_epoch_end(self, trainer: Any, epoch: int):
        """Called at the end of each epoch."""
        pass

    def on_step_end(self, trainer: Any, step: int, epoch: int, loss: float):
        """Called at the end of each training step."""
        pass

class CheckpointCallback(TrainingCallback):
    """Saves model checkpoints during training."""
    
    def __init__(self, checkpoint_dir: Path, save_steps: int):
        self.checkpoint_dir = checkpoint_dir
        self.save_steps = save_steps
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def on_step_end(self, trainer: Any, step: int, epoch: int, loss: float):
        if step > 0 and step % self.save_steps == 0:
            path = self.checkpoint_dir / f"checkpoint-{epoch}-{step}.pt"
            trainer.save_checkpoint(path)

class MetricsLoggingCallback(TrainingCallback):
    """Logs metrics to file during training."""
    
    def __init__(self, log_dir: Path, logger: Optional[logging.Logger] = None):
        self.log_dir = log_dir
        self.logger = logger or logging.getLogger(__name__)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.start_time = None

    def on_training_start(self, trainer: Any):
        self.start_time = time.perf_counter()
        self._log_event("training_started")

    def on_training_end(self, trainer: Any, training_time: float):
        metrics = trainer.metrics.get_latest_metrics()
        self._log_event("training_completed", {
            "total_time": training_time,
            "final_metrics": metrics
        })

    def _log_event(self, event: str, extra: Dict[str, Any] = None):
        log_data = {
            "event": event,
            "timestamp": time.time(),
            **(extra or {})
        }
        log_path = self.log_dir / f"{event}.json"
        with open(log_path, "w") as f:
            json.dump(log_data, f, indent=2)
