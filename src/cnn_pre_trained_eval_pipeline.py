"""Module for evaluating pre-trained summarization models.

This module provides functionality to evaluate pre-trained models from 
Hugging Face on test datasets without needing to train them first.
"""

from all_dataclass import LoadDatasetResult, Metrics, PreprocessedDataset, LoadPretrainedModelResult, PretrainedEvalResult
from data_processing.load_and_save_dataset import load_and_save_dataset
from data_processing.load_tokenizer import load_tokenizer
from data_processing.preprocess_dataset import preprocess_dataset
from model.evaluate_model import evaluate_model
from model.load_pretrained_model import load_pretrained_model
from model.save_evaluation_results import save_evaluation_results
from utils.load_config import load_config

import os
from typing import Any

def run_pretrained_eval_pipeline(
    config_path: str,
    output_dir: str,
    logger: Any
) -> PretrainedEvalResult:
    """Run the evaluation pipeline using a pre-trained model.

    Args:
        config_path: Path to configuration file
        output_dir: Directory to store outputs
        logger: Logger instance for logging

    Returns:
        PretrainedEvalResult: The result of the evaluation pipeline, including
        the loaded model and evaluation metrics.
    """
    logger.info("Running pre-trained model evaluation pipeline.")
    config = load_config(config_path=config_path, logger=logger)

    loaded_dataset: LoadDatasetResult = load_and_save_dataset(
        sample_size=config.dataset.sample_size,
        dataset=config.dataset,
        logger=logger
    )
    tokenizer = load_tokenizer(
        model_name=config.model.model_name,
        logger=logger
    )
    preprocessed_dataset: PreprocessedDataset = preprocess_dataset(
        raw_dataset=loaded_dataset.dataset,
        dataset_config=config.dataset,
        tokenizer=tokenizer,
        logger=logger
    )

    # Load pre-trained model
    pretrained_model_result: LoadPretrainedModelResult = load_pretrained_model(
        model_name=config.model.pretrained_model_name,
        logger=logger
    )

    # Create a wrapper class that matches the expected format for evaluate_model
    class ModelWrapper:
        def __init__(self, model):
            self.model = model

    train_model_result = ModelWrapper(pretrained_model_result.model)

    # Evaluate the pre-trained model
    evaluation_result: Metrics = evaluate_model(
        train_model_result=train_model_result,
        dataset=preprocessed_dataset,
        tokenizer=tokenizer,
        logger=logger
    )

    # Save the evaluation results to the output directory
    save_results = save_evaluation_results(
        metrics=evaluation_result,
        output_dir=output_dir,
        logger=logger
    )

    return PretrainedEvalResult(
        model_result=pretrained_model_result,
        evaluation_metrics=evaluation_result,
        save_results=save_results
    )
