"""
SageMaker training script for BART summarization models.
Handles data loading, training, and model saving in SageMaker environment.
"""

import os
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
import transformers
from transformers import set_seed
from datasets import Dataset

from ..data_processing.huggingface_loader import HuggingFaceLoader
from ..data_processing.dataset_config import DATASET_CONFIGS
from ..models.abstractive_sum.bart_trainer import train_bart_model, TrainingMetrics
from ..models.abstractive_sum.bart_abstractive_model import BartModel, BartConfig
from ..utils.project_logger import get_logger


@dataclass
class SageMakerTrainingConfig:
    """SageMaker training configuration parameters."""
    model_name: str
    dataset_name: str
    dataset_split: str
    output_dir: Path
    num_epochs: int
    batch_size: int
    learning_rate: float
    max_length: int
    min_length: int
    warmup_steps: int
    weight_decay: float
    seed: int = 42


def parse_sagemaker_args() -> SageMakerTrainingConfig:
    """Parse SageMaker training arguments."""
    parser = argparse.ArgumentParser()
    
    # SageMaker specific arguments
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--output-data-dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR"))
    
    # Training arguments
    parser.add_argument("--model-name", type=str, default="facebook/bart-large-cnn")
    parser.add_argument("--dataset-name", type=str, default="cnn_dailymail")
    parser.add_argument("--dataset-split", type=str, default="train")
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--min-length", type=int, default=50)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    return SageMakerTrainingConfig(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        output_dir=Path(args.model_dir),
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        min_length=args.min_length,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        seed=args.seed
    )


def train_model_on_sagemaker() -> None:
    """Execute training process on SageMaker."""
    # Get configuration
    config = parse_sagemaker_args()
    
    # Set up logging
    logger = get_logger("sagemaker_training", level="INFO")
    logger.info(f"Starting SageMaker training: {vars(config)}")
    
    # Set seed for reproducibility
    set_seed(config.seed)
    
    # Initialize model configuration
    model_config = BartConfig(
        model_name=config.model_name,
        max_length=config.max_length,
        min_length=config.min_length,
        num_beams=4,
        device_map="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Initialize the model
    bart_model = BartModel(config=model_config)
    
    # Load and prepare dataset
    loader = HuggingFaceLoader.from_preset(
        dataset_name=config.dataset_name, # type: ignore
        tokenizer=bart_model.tokenizer,  # Set tokenizer from initialized model
        max_length=config.max_length,
        batch_size=config.batch_size
    )
    
    # Load datasets
    train_dataset = loader.load_and_preprocess(
        split=config.dataset_split,
        max_length=config.max_length,
        truncation=True,
        padding=True
    )
    
    val_dataset = loader.load_and_preprocess(
        split="validation",
        max_length=config.max_length,
        truncation=True,
        padding=True
    )
    
    # Execute training using the functional approach from bart_trainer.py
    logger.info("Beginning model training")
    train_results = train_bart_model(
        model=bart_model,
        train_dataset=train_dataset.dataset, # type: ignore
        val_dataset=val_dataset.dataset, # type: ignore
        batch_size=config.batch_size,
        num_epochs=config.num_epochs,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        weight_decay=config.weight_decay,
        logger=logger
    )
    
    # Save model
    output_path = config.output_dir / "model"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save the model directly
    bart_model.model.save_pretrained(output_path)
    bart_model.tokenizer.save_pretrained(output_path)
    
    # Save a config file with version information
    with open(output_path / "version.txt", "w") as f:
        f.write("version: 1.0.0\n")
    
    logger.info(f"Model saved to {output_path}, metrics: {train_results}")


if __name__ == "__main__":
    train_model_on_sagemaker()