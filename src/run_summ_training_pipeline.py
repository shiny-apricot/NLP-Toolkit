from all_dataclass import LoadDatasetResult, Metrics, PipelineResult, PreprocessedDataset, TrainModelResult
from data_processing.load_and_save_dataset import load_and_save_dataset
from data_processing.load_tokenizer import load_tokenizer
from data_processing.preprocess_dataset import preprocess_dataset
from model.evaluate_model import evaluate_model
from model.save_evaluation_results import save_evaluation_results
from model.train_model import train_model
from utils.load_config import load_config

import os
from typing import Any

import yaml
import argparse
from pathlib import Path
from utils.create_timestamped_output_dir import create_timestamped_output_dir
from run_summ_training_pipeline import run_summarization_pipeline
from utils.project_logger import setup_logger


def run_summarization_pipeline(
    config_path: str,
    output_dir: str,
    logger: Any
) -> PipelineResult:
    """Run the full text summarization pipeline.

    Args:
        config_path: Path to configuration file
        output_dir: Directory to store outputs
        logger: Logger instance for logging.

    Returns:
        PipelineResult: The result of the summarization pipeline, including
        the trained model path and evaluation metrics.
    """
    logger.info("Running summarization pipeline.")
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

    # Use timestamped output directory for model outputs
    model_output_dir = os.path.join(output_dir, "model")
    os.makedirs(model_output_dir, exist_ok=True)

    train_model_result: TrainModelResult = train_model(
        dataset=preprocessed_dataset,
        model_name=config.model.model_name,
        output_dir=model_output_dir,
        save_model=config.output.save_model,
        logger=logger
    )
    evaluation_result: Metrics = evaluate_model(
        train_model_result=train_model_result,
        dataset=preprocessed_dataset,
        tokenizer=tokenizer,
        logger=logger
    )

    # Save the evaluation results to the timestamped directory
    save_results = save_evaluation_results(
        metrics=evaluation_result,
        output_dir=output_dir,
        logger=logger
    )

    return PipelineResult(
        train_model_result=train_model_result,
        evaluation_metrics=evaluation_result,
        save_results=save_results
    )



if __name__ == "__main__":
    """Main entry point for the text summarization pipeline.
    
    Sets up logging, executes the complete summarization pipeline,
    and reports success or failure.
    """
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description='Run text summarization pipeline')
    parser.add_argument('--config', type=str, default='test_config.yaml',
                        help='Configuration filename (stored in ./configs/ directory)')
    args = parser.parse_args()
    
    print("Starting text summarization pipeline...")
    
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
    logger = setup_logger('summarization.log', output_dir=timestamped_dir)
    logger.info(f"Text summarization pipeline initialized. Outputs will be saved to: {timestamped_dir}")
    
    try:
        result = run_summarization_pipeline(
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
        print(f"Text summarization pipeline completed successfully. Outputs saved to: {timestamped_dir}")