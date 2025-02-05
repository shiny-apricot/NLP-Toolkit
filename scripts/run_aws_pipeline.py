"""AWS deployment entry point for summarization pipeline"""

import argparse
from pathlib import Path
import torch
import os
from datetime import datetime

from src.utils.project_logger import ProjectLogger
from src.utils.aws_resource_manager import AWSResourceManager
from src.utils.config_loader import load_instance_config
from src.summarization_pretrained_pipeline import run_complete_pipeline
from src.utils.aws_instance_manager import InstanceManager
from src.utils.memory_utils import MemoryTracker, optimize_memory

def setup_aws_environment(
    *,
    config_path: Path,
    logger: ProjectLogger
) -> tuple[AWSResourceManager, dict]:
    """Setup AWS resources and environment variables from config."""
    # Load instance configuration
    instance_config = load_instance_config(config_path)
    
    resource_manager = AWSResourceManager(
        region=instance_config.aws.region,
        instance_type=instance_config.instance_type,
        logger=logger
    )
    
    # Configure environment for optimal AWS performance
    os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5"
    os.environ["CUDA_VISIBLE_DEVICES"] = instance_config.compute.cuda_visible_devices
    torch.set_num_threads(instance_config.compute.torch_num_threads)
    
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    
    return resource_manager, instance_config

def main():
    parser = argparse.ArgumentParser(description="Run summarization pipeline on AWS")
    parser.add_argument(
        "--config", 
        type=Path, 
        default=Path("configs/aws_configs/p3.2xlarge.yaml"),
        help="Path to instance configuration YAML"
    )
    parser.add_argument("--dataset-split", choices=["train", "validation", "test"], default="validation")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument(
        "--stop-instance",
        choices=['no', 'stop', 'terminate'],
        default='no',
        help="Whether to stop/terminate instance after completion"
    )
    args = parser.parse_args()

    # Initialize logger with AWS-specific configuration
    logger = ProjectLogger(
        name="aws_summarization",
        log_path=Path(f"logs/aws_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    )
    
    instance_manager = None
    if args.stop_instance != 'no':
        instance_manager = InstanceManager(logger)

    try:
        with MemoryTracker(logger, "Pipeline Execution", log_path=Path("logs/memory_tracking.jsonl")):
            # Setup AWS environment and resources using config
            resource_manager, instance_config = setup_aws_environment(
                config_path=args.config,
                logger=logger
            )

            with resource_manager:
                # Run the pipeline with config parameters
                results = run_complete_pipeline(
                    model_name=instance_config.model.model_name,
                    dataset_split=args.dataset_split,
                    sample_size=args.sample_size,
                    max_length=instance_config.training.max_length,
                    batch_size=instance_config.training.batch_size,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    logger=logger
                )

                # Optimize memory after major operations
                optimize_memory(logger=logger)

                # Log results to CloudWatch using config settings
                logger.info(
                    "Pipeline completed successfully",
                    metrics=results.evaluation_metrics.to_dict(),
                    resource_usage=resource_manager.get_resource_usage(),
                    config=instance_config
                )

                # After pipeline completion, handle instance stopping
                if instance_manager:
                    logger.info("Pipeline completed, preparing to stop instance")
                    try:
                        if args.stop_instance == 'stop':
                            instance_manager.stop_instance()
                        elif args.stop_instance == 'terminate':
                            instance_manager.terminate_instance()
                    except Exception as e:
                        logger.error(f"Failed to stop instance: {e}")

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise
    finally:
        logger.info("Cleaning up resources")

if __name__ == "__main__":
    main()
