"""
Module for deploying trained summarization models to SageMaker endpoints.
Handles model packaging, deployment configuration, and endpoint creation.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, Union
from pathlib import Path
import boto3
import sagemaker
from sagemaker.pytorch import PyTorchModel

from ..utils.project_logger import get_logger


@dataclass
class DeploymentConfig:
    """Configuration for model deployment."""
    model_s3_path: str
    instance_type: str
    endpoint_name: str
    role_arn: str
    pytorch_version: str = "1.13.1"
    py_version: str = "py39"
    instance_count: int = 1
    env_vars: Optional[Dict[str, str]] = None


def deploy_summarization_endpoint(
    *,
    model_s3_path: str,
    instance_type: str,
    endpoint_name: str,
    role_arn: Optional[str] = None,
    region_name: str = "us-west-2",
    logger: Optional[Any] = None
) -> str:
    """
    Deploy a summarization model to a SageMaker endpoint.
    
    Args:
        model_s3_path: S3 URI to the model.tar.gz file
        instance_type: SageMaker instance type for deployment
        endpoint_name: Name for the SageMaker endpoint
        role_arn: IAM role ARN for SageMaker (uses default role if not specified)
        region_name: AWS region for deployment
        logger: Optional logger instance
        
    Returns:
        Name of the created endpoint
    """
    if logger is None:
        logger = get_logger("model_deployment", level="INFO")
    
    # Initialize SageMaker session
    boto_session = boto3.Session(region_name=region_name)
    sagemaker_session = sagemaker.Session(boto_session=boto_session)
    
    # Get SageMaker role ARN if not provided
    if role_arn is None:
        role_arn = sagemaker.get_execution_role(sagemaker_session)
        logger.info(f"Using default SageMaker execution role: {role_arn}")
    
    # Environment variables for the model container
    env_vars = {
        "MODEL_CACHE_ROOT": "/opt/ml/model",
        "TRANSFORMERS_CACHE": "/opt/ml/model",
        "SAGEMAKER_CONTAINER_LOG_LEVEL": "20"  # INFO level
    }
    
    # Create PyTorch model
    model = PyTorchModel(
        model_data=model_s3_path,
        role=role_arn,
        entry_point="inference.py",  # This script must exist in the model.tar.gz
        source_dir=None,
        framework_version="1.13.1",
        py_version="py39",
        env=env_vars,
        sagemaker_session=sagemaker_session
    )
    
    # Deploy the model
    logger.info(f"Deploying model to endpoint: {endpoint_name}")
    predictor = model.deploy(
        initial_instance_count=1,
        instance_type=instance_type,
        endpoint_name=endpoint_name
    )
    
    logger.info(f"Endpoint {endpoint_name} deployed successfully")
    return endpoint_name
