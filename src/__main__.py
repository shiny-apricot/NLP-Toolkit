"""Main module for orchestrating the text summarization pipeline.

This module defines the high-level workflow for the text summarization project.
"""

import yaml
from utils.create_timestamped_output_dir import create_timestamped_output_dir
from run_summarization_pipeline import run_summarization_pipeline
from utils.project_logger import setup_logger


if __name__ == "__main__":
    """Main entry point for the text summarization pipeline.
    
    Sets up logging, executes the complete summarization pipeline,
    and reports success or failure.
    """
    print("Starting text summarization pipeline...")
    
    # Load base config to get output directory
    with open("./configs/test_config.yaml", "r") as file:
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
            config_path="./configs/test_config.yaml", 
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