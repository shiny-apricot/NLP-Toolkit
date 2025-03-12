"""BART model training functionality."""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Callable
import logging
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup
from .bart_abstractive_model import BartModel

class SummarizationError(Exception):
    """Base class for summarization errors."""
    pass

class TrainingError(SummarizationError):
    """Error encountered during model training."""
    pass

@dataclass
class TrainingMetrics:
    """Training result metrics."""
    train_loss: float
    val_loss: Optional[float] = None
    final_learning_rate: Optional[float] = None

def create_optimizer(
    model: BartModel,
    *,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01
) -> torch.optim.Optimizer:
    """Initialize optimizer with weight decay configuration.
    
    Args:
        model: BART model instance
        learning_rate: Learning rate for training
        weight_decay: Weight decay factor for regularization
        
    Returns:
        Configured optimizer
    """
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_params = [
        {
            "params": [p for n, p in model.model.named_parameters() 
                      if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.model.named_parameters() 
                      if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    return torch.optim.AdamW(optimizer_params, lr=learning_rate)

def create_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    train_dataset_size: int,
    batch_size: int,
    num_epochs: int,
    warmup_steps: int = 500
) -> torch.optim.lr_scheduler.LambdaLR:
    """Initialize learning rate scheduler.
    
    Args:
        optimizer: Training optimizer
        train_dataset_size: Size of the training dataset
        batch_size: Batch size used in training
        num_epochs: Number of training epochs
        warmup_steps: Number of warmup steps
        
    Returns:
        Learning rate scheduler
    """
    steps_per_epoch = max(1, train_dataset_size // batch_size)
    num_training_steps = steps_per_epoch * num_epochs
    
    return get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=min(warmup_steps, num_training_steps // 10),
        num_training_steps=num_training_steps
    )

def training_step(
    model: BartModel,
    batch: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: Optional[torch.cuda.amp.GradScaler],
    *,
    step: int,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float = 1.0,
    use_bfloat16: bool = False
) -> torch.Tensor:
    """Execute single training step.
    
    Args:
        model: BART model instance
        batch: Training batch data
        optimizer: Model optimizer
        scheduler: Learning rate scheduler
        scaler: Gradient scaler for mixed precision training
        step: Current training step
        gradient_accumulation_steps: Steps for gradient accumulation
        max_grad_norm: Maximum gradient norm for clipping
        use_bfloat16: Whether to use bfloat16 precision
        
    Returns:
        Loss tensor
    """
    # Move batch to the same device as the model
    batch = {k: v.to(model.model.device) for k, v in batch.items()}
    
    # Zero gradients at the beginning if we're starting a new accumulation cycle
    if step % gradient_accumulation_steps == 0:
        optimizer.zero_grad()
    
    with torch.cuda.amp.autocast(enabled=use_bfloat16):
        outputs = model.model(**batch)
        loss = outputs.loss / gradient_accumulation_steps
    
    if scaler is not None:
        scaler.scale(loss).backward()
        if (step + 1) % gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.model.parameters(),
                max_grad_norm
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
    else:
        loss.backward()
        if (step + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(
                model.model.parameters(),
                max_grad_norm
            )
            optimizer.step()
            scheduler.step()
        
    return loss

def validate_model(
    model: BartModel,
    val_dataset: Dataset,
    *,
    batch_size: int = 8,
    use_bfloat16: bool = False,
    num_workers: int = 2,
    logger: Optional[logging.Logger] = None
) -> float:
    """Run validation on the model.
    
    Args:
        model: BART model instance
        val_dataset: Validation dataset
        batch_size: Validation batch size
        use_bfloat16: Whether to use bfloat16 precision
        num_workers: Number of worker processes for data loading
        logger: Optional logger instance
        
    Returns:
        Average validation loss
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    model.model.eval()
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    total_loss = 0
    num_batches = 0
    
    try:
        with torch.no_grad():
            for batch in val_dataloader:
                # Move batch to the same device as the model
                batch = {k: v.to(model.model.device) for k, v in batch.items()}
                
                with torch.cuda.amp.autocast(enabled=use_bfloat16):
                    outputs = model.model(**batch)
                    total_loss += outputs.loss.item()
                
                num_batches += 1
                
        model.model.train()
        return total_loss / max(num_batches, 1)  # Avoid division by zero
    except Exception as e:
        logger.error(f"Validation error: {str(e)}")
        model.model.train()
        return float('inf')

def save_checkpoint(
    model: BartModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    *,
    step: int,
    output_path: str = 'checkpoint-{step}.pt'
) -> None:
    """Save training checkpoint.
    
    Args:
        model: BART model instance
        optimizer: Model optimizer
        scheduler: Learning rate scheduler
        step: Current training step
        output_path: Path template for saving checkpoint
    """
    try:
        # Format the output path with the step number
        formatted_path = output_path.format(step=step)
    except KeyError:
        # Fallback if formatting fails
        formatted_path = f"checkpoint-{step}.pt"
        
    checkpoint = {
        'model_state_dict': model.model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'step': step,
    }
    torch.save(checkpoint, formatted_path)

def train_bart_model(
    model: BartModel,
    train_dataset: Dataset,
    *,
    val_dataset: Optional[Dataset] = None,
    batch_size: int = 8,
    num_epochs: int = 3,
    learning_rate: float = 2e-5,
    warmup_steps: int = 500,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float = 1.0,
    weight_decay: float = 0.01,
    save_steps: int = 1000,
    use_bfloat16: bool = False,
    num_workers: int = 2,
    logger: Optional[logging.Logger] = None
) -> TrainingMetrics:
    """Execute BART model training.
    
    Args:
        model: BART model instance
        train_dataset: Training dataset
        val_dataset: Optional validation dataset
        batch_size: Training batch size
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        warmup_steps: Warmup steps for scheduler
        gradient_accumulation_steps: Steps for gradient accumulation
        max_grad_norm: Maximum gradient norm for clipping
        weight_decay: Weight decay for optimizer
        save_steps: How often to save checkpoints
        use_bfloat16: Whether to use bfloat16 precision
        num_workers: Number of worker processes for data loading
        logger: Optional logger instance
    
    Returns:
        TrainingMetrics with training results
        
    Raises:
        TrainingError: If an error occurs during training
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        logger.addHandler(console_handler)

    try:
        model.model.train()
        optimizer = create_optimizer(
            model, 
            learning_rate=learning_rate, 
            weight_decay=weight_decay
        )
        
        # Setup gradient scaler for mixed precision training
        scaler = torch.cuda.amp.GradScaler() if use_bfloat16 else None
        
        # Get dataset size safely, fallback to estimating from dataloader if __len__ not available
        try:
            train_dataset_size = len(train_dataset) # type: ignore
        except (TypeError, AttributeError):
            # Create a temporary dataloader to estimate size
            temp_loader = DataLoader(train_dataset, batch_size=1)
            train_dataset_size = len(temp_loader) * batch_size
            logger.warning("Dataset has no length, estimating size from dataloader")
            
        scheduler = create_scheduler(
            optimizer,
            train_dataset_size=train_dataset_size,
            batch_size=batch_size,
            num_epochs=num_epochs,
            warmup_steps=warmup_steps
        )

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        total_loss = 0
        global_step = 0
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            
            for step, batch in enumerate(train_dataloader):
                loss = training_step(
                    model, batch, optimizer, scheduler,
                    scaler,
                    step=global_step,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    max_grad_norm=max_grad_norm,
                    use_bfloat16=use_bfloat16
                )
                epoch_loss += loss.item()
                global_step += 1
                
                if (global_step > 0) and (global_step % save_steps == 0):
                    save_checkpoint(model, optimizer, scheduler, step=global_step)
            
            avg_epoch_loss = epoch_loss / len(train_dataloader)
            
            if val_dataset:
                val_loss = validate_model(
                    model, 
                    val_dataset,
                    batch_size=batch_size,
                    use_bfloat16=use_bfloat16,
                    num_workers=num_workers,
                    logger=logger
                )
                logger.info(f"Epoch {epoch}: train_loss={avg_epoch_loss:.4f}, val_loss={val_loss:.4f}")
            else:
                logger.info(f"Epoch {epoch}: train_loss={avg_epoch_loss:.4f}")
            
            total_loss += avg_epoch_loss
        
        # Get final learning rate safely
        final_lr = None
        try:
            final_lr = scheduler.get_last_lr()[0]
        except (AttributeError, IndexError):
            logger.warning("Could not retrieve final learning rate")
        
        final_metrics = TrainingMetrics(
            train_loss=total_loss / num_epochs,
            final_learning_rate=final_lr
        )
        
        if val_dataset:
            final_metrics.val_loss = validate_model(
                model, 
                val_dataset,
                batch_size=batch_size,
                use_bfloat16=use_bfloat16,
                num_workers=num_workers,
                logger=logger
            )
            
        return final_metrics
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise TrainingError(f"Error during model training: {str(e)}") from e
