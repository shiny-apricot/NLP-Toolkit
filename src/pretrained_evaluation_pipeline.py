"""Main module for evaluating pre-trained summarization models.

This module defines the workflow for evaluating pre-trained summarization models
on test datasets without training.
"""

import yaml
import argparse
from pathlib import Path
from utils.create_timestamped_output_dir import create_timestamped_output_dir
from cnn_pre_trained_eval_pipeline import run_pretrained_eval_pipeline
from utils.project_logger import setup_logger


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
