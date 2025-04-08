"""Main module for orchestrating the text summarization pipeline.

This module defines the high-level workflow for the text summarization project.
"""

from dataclasses import dataclass
import yaml
from typing import Any, Dict, List, Optional
from utils.project_logger import get_logger
from all_dataclass import (
    Config, PipelineResult, DatasetConfig, ModelConfig, TrainingConfig, 
    EvaluationConfig, OutputConfig, LoadDatasetResult, PreprocessedDataset, 
    TrainModelResult, Metrics, EvaluationResult
)
from datasets import load_dataset
import os



def load_config(config_path: str, logger: Any) -> Config:
    """Load the configuration file into a Config dataclass.

    Args:
        config_path: Path to the YAML configuration file.
        logger: Logger instance for logging.

    Returns:
        Config: Parsed configuration as a Config dataclass.
    """
    logger.info(f"Loading configuration from {config_path}.")
    with open(config_path, "r") as file:
        raw_config = yaml.safe_load(file)
    return Config(
        dataset=DatasetConfig(**raw_config["dataset"]),
        model=ModelConfig(**raw_config["model"]),
        training=TrainingConfig(**raw_config["training"]),
        evaluation=EvaluationConfig(**raw_config["evaluation"]),
        output=OutputConfig(**raw_config["output"]),
    )

def run_summarization_pipeline(logger: Any) -> PipelineResult:
    """Run the full text summarization pipeline.

    Args:
        logger: Logger instance for logging.

    Returns:
        PipelineResult: The result of the summarization pipeline, including
        the trained model path and evaluation metrics.
    """
    logger.info("Running summarization pipeline.")
    config = load_config(config_path="./configs/test_config.yaml", logger=logger)
    loaded_dataset = load_and_save_dataset(
        sample_size=config.dataset.sample_size,
        dataset=config.dataset,
        logger=logger
    )
    preprocessed_dataset = preprocess_dataset(
        raw_dataset=loaded_dataset.dataset,
        tokenizer=loaded_dataset.tokenizer,
        logger=logger
    )
    train_model_result = train_model(
        dataset=preprocessed_dataset,
        model_type=config.model.model_type,
        logger=logger
    )
    evaluation_result = evaluate_model(
        train_model_result=train_model_result,
        dataset=loaded_dataset,
        logger=logger
    )
    return PipelineResult(
        train_model_result=train_model_result,
        evaluation_metrics=evaluation_result
    )

def load_and_save_dataset(sample_size: int, dataset: DatasetConfig, logger: Any) -> LoadDatasetResult:
    """Load and preprocess the dataset from Hugging Face and save it locally.

    Args:
        sample_size: Number of samples to load.
        dataset: Dataset configuration.
        logger: Logger instance for logging.

    Returns:
        LoadDatasetResult: Dataclass containing dataset and tokenizer.
    """
    logger.info(f"Loading dataset {dataset.dataset_name} with sample size {sample_size}.")
    
    data_dir = os.path.join(os.path.dirname(__file__), "../data")
    os.makedirs(data_dir, exist_ok=True)
    dataset_path = os.path.join(data_dir, f"{dataset.dataset_name}.arrow")

    hf_dataset = load_dataset(dataset.dataset_name, dataset.dataset_version)

    result = LoadDatasetResult(
        dataset=hf_dataset,
        tokenizer=None,
        train_dataset=hf_dataset["train"].select(range(sample_size)),  # type: ignore
        test_dataset=hf_dataset["test"].select(range(sample_size)),  # type: ignore
        val_dataset=hf_dataset["validation"].select(range(sample_size))  # type: ignore
    )
    return result

def preprocess_dataset(
    raw_dataset: Any,
    tokenizer: Any,
    logger: Any
) -> PreprocessedDataset:
    """Preprocess the raw dataset for training.

    Args:
        raw_dataset: The raw dataset to preprocess.
        tokenizer: The tokenizer to use for preprocessing.
        logger: Logger instance for logging.

    Returns:
        PreprocessedDataset: Dataclass containing preprocessed datasets.
    """
    logger.info("Preprocessing dataset.")
    # Example preprocessing logic
    train_dataset = raw_dataset["train"].map(lambda x: tokenizer(x["text"]))
    test_dataset = raw_dataset["test"].map(lambda x: tokenizer(x["text"]))
    val_dataset = raw_dataset["validation"].map(lambda x: tokenizer(x["text"]))

    return PreprocessedDataset(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        val_dataset=val_dataset
    )

def train_model(dataset: Any, model_type: str, logger: Any) -> TrainModelResult:
    """Train the summarization model.

    Args:
        dataset: The dataset to use for training.
        model_type: The type of model to train (e.g., 'bart').
        logger: Logger instance for logging.

    Returns:
        TrainModelResult: Dataclass containing the trained model and related info.
    """
    logger.info(f"Training model of type {model_type}.")
    result = TrainModelResult(
        model=None,
        tokenizer=None,
        training_args=None
    )
    return result

def evaluate_model(train_model_result: TrainModelResult, dataset: Any, logger: Any) -> Metrics:
    """Evaluate the trained model.

    Args:
        train_model_result: Result of the trained model.
        dataset: The dataset to use for evaluation.
        logger: Logger instance for logging.

    Returns:
        Metrics: Evaluation metrics.
    """
    logger.info("Evaluating the model.")
    result = Metrics()
    return result

if __name__ == "__main__":
    """Main function to run the text summarization pipeline."""
    print("Starting text summarization pipeline...")
    logger = get_logger(__name__)
    logger.info("Starting text summarization pipeline.")
    result = run_summarization_pipeline(logger=logger)
    logger.info("Pipeline completed successfully.")