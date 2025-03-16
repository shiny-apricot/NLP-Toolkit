"""
Script to launch BART training jobs on Amazon SageMaker.
Handles job configuration, IAM roles, and model artifacts.
"""

from dataclasses import dataclass
import os
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, List
import boto3
import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.local import LocalSession

from ..utils.project_logger import get_logger
from ..utils.model_storage import format_input_path, get_default_model_location, parse_storage_location


@dataclass
class SageMakerJobConfig:
    """Configuration for SageMaker training job."""
    job_name: str
    role_arn: str
    instance_type: str
    instance_count: int
    pytorch_version: str = "1.13.1"
    py_version: str = "py39"
    volume_size: int = 100
    max_run_seconds: int = 86400  # 24 hours
    data_dir: str = "/tmp/summarization-data"
    output_path: Optional[str] = None
    model_name: str = "facebook/bart-large-cnn"
    dataset_name: str = "cnn_dailymail"
    use_s3: bool = False
    s3_bucket: Optional[str] = None
    hyperparameters: Optional[Dict[str, Any]] = None


def prepare_source_dir(
    *,
    source_dir: Path,
    entry_point: Path
) -> Path:
    """
    Prepare source directory for SageMaker training.
    
    Args:
        source_dir: Base directory for source code
        entry_point: Entry point script path
        
    Returns:
        Path to prepared directory
    """
    # Ensure directories exist
    source_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy necessary files if needed
    # (In a real project, you might need to package dependencies here)
    
    return source_dir


def launch_training_job(
    *,
    config: SageMakerJobConfig,
    source_dir: Path,
    entry_point: Path,
    logger: Optional[Any] = None
) -> str:
    """
    Launch SageMaker training job.
    
    Args:
        config: Job configuration
        source_dir: Source directory
        entry_point: Entry point script
        logger: Optional logger
        
    Returns:
        Training job name
    """
    if logger is None:
        logger = get_logger("sagemaker_launcher", level="INFO")
    
    # Create local SageMaker session if using local instance
    if config.instance_type.startswith("local"):
        sagemaker_session = LocalSession()
        sagemaker_session.config = {'local': {'local_code': True}}
    else:
        sagemaker_session = sagemaker.Session()
    
    # Set default output path if not provided
    if not config.output_path:
        config.output_path = get_default_model_location(
            model_name=config.job_name,
            use_s3=config.use_s3,
            s3_bucket=config.s3_bucket
        )
    
    logger.info(f"Model output will be stored at: {config.output_path}")
    
    # Prepare hyperparameters
    hyperparameters = config.hyperparameters or {
        "model-name": config.model_name,
        "dataset-name": config.dataset_name,
        "num-epochs": 3,
        "batch-size": 4,
        "learning-rate": 2e-5,
        "max-length": 512,
        "min-length": 50,
    }
    
    # Format paths correctly for SageMaker
    formatted_output = format_input_path(config.output_path)
    
    # Create SageMaker estimator
    estimator = PyTorch(
        entry_point=entry_point.name,
        source_dir=str(source_dir),
        role=config.role_arn,
        instance_count=config.instance_count,
        instance_type=config.instance_type,
        framework_version=config.pytorch_version,
        py_version=config.py_version,
        max_run=config.max_run_seconds,
        volume_size=config.volume_size,
        hyperparameters=hyperparameters,
        output_path=formatted_output,
        sagemaker_session=sagemaker_session
    )
    
    logger.info(f"Launching SageMaker training job: {config.job_name}")
    
    # Format data directory correctly
    formatted_data_dir = format_input_path(config.data_dir)
    
    # Use formatted path for training
    estimator.fit({'training': formatted_data_dir}, job_name=config.job_name)
    
    logger.info(f"Training job completed. Model artifacts: {estimator.model_data}")
    return config.job_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch SageMaker training job")
    parser.add_argument("--job-name", type=str, required=True, help="Training job name")
    parser.add_argument("--role-arn", type=str, required=True, help="IAM role ARN for SageMaker")
    parser.add_argument("--instance-type", type=str, default="local", help="Instance type")
    parser.add_argument("--instance-count", type=int, default=1, help="Number of instances")
    parser.add_argument("--model-name", type=str, default="facebook/bart-large-cnn", help="Model name")
    parser.add_argument("--data-dir", type=str, default="/tmp/summarization-data", help="Data directory")
    parser.add_argument("--output-path", type=str, help="Output directory")
    parser.add_argument("--use-s3", action="store_true", help="Use S3 for storage")
    parser.add_argument("--s3-bucket", type=str, help="S3 bucket name for storage")
    
    args = parser.parse_args()
    
    config = SageMakerJobConfig(
        job_name=args.job_name,
        role_arn=args.role_arn,
        instance_type=args.instance_type,
        instance_count=args.instance_count,
        model_name=args.model_name,
        data_dir=args.data_dir,
        output_path=args.output_path,
        use_s3=args.use_s3,
        s3_bucket=args.s3_bucket
    )
    
    # Use the existing sagemaker_training.py as the entry point
    source_dir = Path(__file__).parent  # Points to the src/training directory
    entry_point = Path("sagemaker_training.py")  # Direct file name 
    
    launch_training_job(
        config=config,
        source_dir=source_dir,
        entry_point=entry_point
    )