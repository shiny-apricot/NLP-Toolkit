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

from ..utils.project_logger import get_logger  # Fix relative import


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
    bucket: str = "summarization-models"
    output_path: Optional[str] = None
    model_name: str = "facebook/bart-large-cnn"
    dataset_name: str = "cnn_dailymail"
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
    
    # Create SageMaker session
    boto_session = boto3.Session(region_name="us-east-1")
    sagemaker_session = sagemaker.Session(boto_session=boto_session)
    
    # Set default output path if not provided
    output_path = config.output_path or f"s3://{config.bucket}/models/{config.job_name}"
    
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
        output_path=output_path,
        sagemaker_session=sagemaker_session
    )
    
    logger.info(f"Launching SageMaker training job: {config.job_name}")
    estimator.fit({'training': f"s3://{config.bucket}/datasets"}, job_name=config.job_name)
    
    logger.info(f"Training job completed. Model artifacts: {estimator.model_data}")
    return config.job_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch SageMaker training job")
    parser.add_argument("--job-name", type=str, required=True, help="Training job name")
    parser.add_argument("--role-arn", type=str, required=True, help="IAM role ARN for SageMaker")
    parser.add_argument("--instance-type", type=str, default="ml.p3.2xlarge", help="Instance type")
    parser.add_argument("--instance-count", type=int, default=1, help="Number of instances")
    parser.add_argument("--model-name", type=str, default="facebook/bart-large-cnn", help="Model name")
    
    args = parser.parse_args()
    
    config = SageMakerJobConfig(
        job_name=args.job_name,
        role_arn=args.role_arn,
        instance_type=args.instance_type,
        instance_count=args.instance_count,
        model_name=args.model_name
    )
    
    # Use the existing sagemaker_training.py as the entry point
    source_dir = Path(__file__).parent.parent  # Points to the src directory
    entry_point = Path("training/sagemaker_training.py")  # Relative to source_dir
    
    launch_training_job(
        config=config,
        source_dir=source_dir,
        entry_point=entry_point
    )