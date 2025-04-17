from all_dataclass import Config, PreprocessedDataset, TrainModelResult


from transformers.models.auto.modeling_auto import AutoModelForSeq2SeqLM
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments


import os
import torch
import random
import numpy as np
from time import time
from typing import Any


def train_model(
    dataset: PreprocessedDataset,
    model_name: str,
    output_dir: str,
    config: Config,
    logger: Any
) -> TrainModelResult:
    """Train the summarization model.

    Args:
        dataset: The preprocessed dataset to use for training.
        model_name: The name of the model to train.
        output_dir: Directory to save model outputs
        config: Complete configuration for the pipeline
        logger: Logger instance for logging.

    Returns:
        TrainModelResult: Dataclass containing the trained model and related info.
    """
    logger.info(f"Initializing model: {model_name}.")
    
    # Set up seed for reproducibility if specified
    if config.hardware and config.hardware.seed is not None:
        seed = config.hardware.seed
        logger.info(f"Setting random seed to {seed} for reproducibility")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Set save_strategy based on save_model parameter
    save_strategy = "steps" if config.output.save_model else "no"
    
    # Extract training configuration
    training_config = config.training
    output_config = config.output
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=training_config.learning_rate,
        per_device_train_batch_size=training_config.batch_size,
        per_device_eval_batch_size=config.evaluation.eval_batch_size,
        num_train_epochs=training_config.num_epochs,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        save_total_limit=2 if config.output.save_model else None,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=output_config.logging_steps,
        eval_steps=training_config.eval_steps,
        report_to="none",  # Disable wandb reporting
        save_strategy=save_strategy,
        save_steps=output_config.save_steps,
        fp16=training_config.mixed_precision,  # Enable mixed precision if configured
    )

    logger.info(f"Model saving is {'enabled' if config.output.save_model else 'disabled'}")
    logger.info(f"Using gradient accumulation with {training_config.gradient_accumulation_steps} steps")
    logger.info(f"Mixed precision training is {'enabled' if training_config.mixed_precision else 'disabled'}")
    logger.info("Starting model training.")
    start_time = time()
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset.train_dataset,
        eval_dataset=dataset.val_dataset,
    )
    
    trainer.train()
    training_time = time() - start_time
    logger.info(f"Model training completed in {training_time:.2f} seconds.")

    return TrainModelResult(
        model=model,
        training_args=training_args
    )


class MyTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Fazla key varsa sil
        if "num_items_in_batch" in inputs:
            inputs = {k: v for k, v in inputs.items() if k != "num_items_in_batch"}
        
        outputs = model(**inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss