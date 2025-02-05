"""Test script for t2.xlarge setup verification"""

from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModel
from src.utils.project_logger import get_logger
import boto3

def verify_setup():
    """Verify the setup is working correctly."""
    logger = get_logger("setup_verification")
    
    try:
        # Test CPU setup
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"Number of CPU threads: {torch.get_num_threads()}")
        
        # Test model loading
        model_name = "facebook/bart-base"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        
        # Test tokenization
        text = "This is a test sentence."
        tokens = tokenizer(text, return_tensors="pt")
        
        logger.info("Setup verification completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Setup verification failed: {str(e)}")
        return False

def verify_aws_permissions():
    """Verify AWS permissions are correctly set up."""
    logger = get_logger("aws_verification")
    
    try:
        # Test S3 access
        s3 = boto3.client('s3')
        s3.list_buckets()
        
        # Test EC2 permissions
        ec2 = boto3.client('ec2')
        ec2.describe_instances()
        
        # Test CloudWatch
        cloudwatch = boto3.client('cloudwatch')
        cloudwatch.list_metrics()
        
        logger.info("AWS permissions verified successfully!")
        return True
        
    except Exception as e:
        logger.error(f"AWS permissions verification failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = verify_setup() and verify_aws_permissions()
    exit(0 if success else 1)
