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

import yaml
import argparse
from pathlib import Path
from utils.create_timestamped_output_dir import create_timestamped_output_dir
from utils.project_logger import setup_logger


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



if __name__ == "__main__":
    """Main entry point for the pre-trained model evaluation pipeline.
    
    Sets up logging, executes the evaluation pipeline using a pre-trained model,
    and reports success or failure.
    """
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description='Run pre-trained model evaluation pipeline')
    parser.add_argument('--config', type=str, default='test_config.yaml',
                        help='Configuration filename (stored in ./configs/ directory)')
    args = parser.parse_args()
    
    print("Starting pre-trained model evaluation pipeline...")
    
    # Construct full config path using configs directory with Path
    config_path = Path('./configs') / args.config
    
    # Load base config to get output directory
    with open(config_path, "r") as file:
        raw_config = yaml.safe_load(file)
    
    # Create timestamped output directory
    base_output_dir = raw_config["output"]["output_dir"]
    timestamped_dir = create_timestamped_output_dir(base_output_dir)
    print(f"Created output directory: {timestamped_dir}")
    
    # Setup logger with the timestamped directory
    logger = setup_logger('pretrained_evaluation.log', output_dir=timestamped_dir)
    logger.info(f"Pre-trained model evaluation pipeline initialized. Outputs will be saved to: {timestamped_dir}")
    
    try:
        result = run_pretrained_eval_pipeline(
            config_path=str(config_path), 
            output_dir=timestamped_dir,
            logger=logger
        )
        logger.info(f"Pipeline completed successfully with ROUGE-L score of {result.evaluation_metrics.rouge_L:.4f}")
        if result.save_results.success:
            logger.info(f"Results saved to {result.save_results.file_path}")
        else:
            logger.warning("Pipeline completed but results could not be saved")
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}")
        print(f"Pipeline failed. See logs for details.")
        raise
    else:
        print(f"Pre-trained model evaluation pipeline completed successfully. Outputs saved to: {timestamped_dir}")
