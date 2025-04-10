from all_dataclass import PreprocessedDataset, TrainModelResult


from transformers.models.auto.modeling_auto import AutoModelForSeq2SeqLM
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments


import os
from time import time
from typing import Any


def train_model(
    dataset: PreprocessedDataset,
    model_name: str,
    output_dir: str,
    logger: Any
) -> TrainModelResult:
    """Train the summarization model.

    Args:
        dataset: The preprocessed dataset to use for training.
        model_name: The name of the model to train.
        output_dir: Directory to save model outputs
        logger: Logger instance for logging.

    Returns:
        TrainModelResult: Dataclass containing the trained model and related info.
    """
    logger.info(f"Initializing model: {model_name}.")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=2,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=10,
        report_to="none",  # Disable wandb reporting
    )

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