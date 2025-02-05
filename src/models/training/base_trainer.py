from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Callable
import torch
from torch.utils.data import DataLoader, Dataset
import logging
from pathlib import Path
import time
from ..utils.gpu_utils import GPUManager, GPUConfig
from .callbacks import TrainingCallback
from .metrics import MetricsTracker

@dataclass
class TrainingConfig:
    """Base configuration for model training."""
    batch_size: int = 16
    num_epochs: int = 3
    learning_rate: float = 2e-5
    warmup_steps: int = 500
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    use_fp16: bool = True
    checkpoint_dir: Path = Path("checkpoints")
    log_steps: int = 100
    eval_steps: int = 500
    save_steps: int = 1000

class BaseTrainer:
    """Base trainer class for implementing common training functionality."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset: Dataset,
        config: TrainingConfig,
        *,  # Force keyword arguments
        val_dataset: Optional[Dataset] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        callbacks: Optional[List[TrainingCallback]] = None,
        logger: Optional[logging.Logger] = None
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.callbacks = callbacks or []
        
        # Initialize GPU management
        gpu_config = GPUConfig(prefer_bfloat16=config.use_fp16)
        self.gpu_manager = GPUManager(gpu_config)
        self.device = self.gpu_manager.select_device()
        
        # Setup model on device
        self.model = self.gpu_manager.optimize_model(self.model, self.device)
        
        # Initialize optimizer and scheduler if not provided
        self.optimizer = optimizer or self._create_optimizer()
        self.scheduler = scheduler or self._create_scheduler()
        
        # Initialize metrics tracker
        self.metrics = MetricsTracker()

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create default optimizer."""
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

    def _create_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        """Create default learning rate scheduler."""
        return torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=self.config.num_epochs
        )

    def train(self) -> Dict[str, float]:
        """Execute training loop."""
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        
        start_time = time.perf_counter()
        
        try:
            # Notify callbacks of training start
            for callback in self.callbacks:
                callback.on_training_start(self)
            
            for epoch in range(self.config.num_epochs):
                self._train_epoch(train_dataloader, epoch)
                
                if self.val_dataset:
                    self._validate_epoch(epoch)
                
                # Update scheduler
                self.scheduler.step()
                
                # Notify callbacks of epoch end
                for callback in self.callbacks:
                    callback.on_epoch_end(self, epoch)
            
            training_time = time.perf_counter() - start_time
            
            # Notify callbacks of training end
            for callback in self.callbacks:
                callback.on_training_end(self, training_time)
            
            return self.metrics.get_latest_metrics()
            
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise

    def _train_epoch(self, train_dataloader: DataLoader, epoch: int):
        """Train for one epoch."""
        self.model.train()
        
        for step, batch in enumerate(train_dataloader):
            loss = self._training_step(batch)
            
            if step % self.config.log_steps == 0:
                self.logger.info(f"Epoch {epoch}, Step {step}, Loss: {loss:.4f}")
            
            # Notify callbacks of step end
            for callback in self.callbacks:
                callback.on_step_end(self, step, epoch, loss)

    def _validate_epoch(self, epoch: int):
        """Run validation for current epoch."""
        self.model.eval()
        val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size
        )
        
        total_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                loss = self._validation_step(batch)
                total_loss += loss
                
        avg_loss = total_loss / len(val_dataloader)
        self.metrics.update({"val_loss": avg_loss})
        
        self.logger.info(f"Epoch {epoch}, Validation Loss: {avg_loss:.4f}")

    def _training_step(self, batch: Any) -> float:
        """
        Implement actual training logic here.
        Should be overridden by specific trainers.
        """
        raise NotImplementedError

    def _validation_step(self, batch: Any) -> float:
        """
        Implement validation logic here.
        Should be overridden by specific trainers.
        """
        raise NotImplementedError

    def save_checkpoint(self, path: Path):
        """Save training checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': self.metrics.get_latest_metrics()
        }, path)

    def load_checkpoint(self, path: Path):
        """Load training checkpoint."""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
