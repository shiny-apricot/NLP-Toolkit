"""BART model training functionality."""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup
from .bart_abstractive_model import BartBaseModel, BartConfig


@dataclass
class TrainingConfig:
    """Training-specific configuration."""
    batch_size: int = 8
    num_epochs: int = 3
    learning_rate: float = 2e-5
    warmup_steps: int = 500
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    save_steps: int = 1000
    eval_steps: int = 500


class BartTrainer(BartBaseModel):
    """BART model training implementation."""
    
    def __init__(
        self,
        model_config: BartConfig,
        training_config: TrainingConfig,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None
    ):
        super().__init__(model_config)
        self.training_config = training_config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Initialize optimizer."""
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_params = [
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                "weight_decay": self.training_config.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        return torch.optim.AdamW(
            optimizer_params,
            lr=self.training_config.learning_rate
        )

    def _create_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        """Initialize learning rate scheduler."""
        num_training_steps = len(self.train_dataset) * self.training_config.num_epochs
        return get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.training_config.warmup_steps,
            num_training_steps=num_training_steps
        )

    def train(self) -> Dict[str, float]:
        """Execute training loop."""
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.training_config.batch_size,
            shuffle=True
        )
        
        self.model.train()
        total_loss = 0
        
        for epoch in range(self.training_config.num_epochs):
            epoch_loss = self._train_epoch(train_dataloader)
            
            if self.val_dataset:
                val_loss = self._validate()
                self.logger.info(f"Epoch {epoch}: train_loss={epoch_loss:.4f}, val_loss={val_loss:.4f}")
            else:
                self.logger.info(f"Epoch {epoch}: train_loss={epoch_loss:.4f}")
            
            total_loss += epoch_loss
        
        return {
            "train_loss": total_loss / self.training_config.num_epochs,
            "final_learning_rate": self.scheduler.get_last_lr()[0]
        }

    def _train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch."""
        epoch_loss = 0
        for step, batch in enumerate(dataloader):
            loss = self._training_step(batch, step)
            epoch_loss += loss.item()
            
            if step % self.training_config.save_steps == 0:
                self._save_checkpoint(step)
                
        return epoch_loss / len(dataloader)

    def _training_step(self, batch: Dict[str, torch.Tensor], step: int) -> torch.Tensor:
        """Execute single training step."""
        with torch.cuda.amp.autocast(enabled=self.config.use_bfloat16):
            outputs = self.model(**batch)
            loss = outputs.loss / self.training_config.gradient_accumulation_steps
            
        loss.backward()
        
        if (step + 1) % self.training_config.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.training_config.max_grad_norm
            )
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            
        return loss

    def _validate(self) -> float:
        """Run validation."""
        self.model.eval()
        val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.training_config.batch_size
        )
        
        total_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                with torch.cuda.amp.autocast(enabled=self.config.use_bfloat16):
                    outputs = self.model(**batch)
                    total_loss += outputs.loss.item()
                    
        self.model.train()
        return total_loss / len(val_dataloader)

    def _save_checkpoint(self, step: int):
        """Save training checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'step': step,
        }
        torch.save(checkpoint, f'checkpoint-{step}.pt')
