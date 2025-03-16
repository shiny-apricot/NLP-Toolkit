"""
Module for calling SageMaker endpoints for text summarization.
Provides utilities for sending requests to deployed summarization models.
"""

import json
import boto3
from dataclasses import dataclass
from typing import Dict, Any, Optional
import logging

from ..utils.project_logger import get_logger


@dataclass
class SummarizationRequest:
    """Parameters for summarization request."""
    text: str
    max_length: int = 150
    min_length: int = 50
    num_beams: int = 4
    do_sample: bool = False
    early_stopping: bool = True
    no_repeat_ngram_size: int = 3
    temperature: float = 1.0


def summarize_with_endpoint(
    *,
    text: str,
    endpoint_name: str,
    max_length: int = 150,
    min_length: int = 50,
    region_name: str = "us-west-2",
    logger: Optional[logging.Logger] = None
) -> str:
    """
    Send text to a SageMaker endpoint for summarization.
    
    Args:
        text: Text to summarize
        endpoint_name: Name of the deployed SageMaker endpoint
        max_length: Maximum length of the summary
        min_length: Minimum length of the summary
        region_name: AWS region where the endpoint is deployed
        logger: Optional logger instance
        
    Returns:
        Summarized text
    """
    if logger is None:
        logger = get_logger("summarization_inference", level="INFO")
    
    # Create request payload
    request = SummarizationRequest(
        text=text,
        max_length=max_length,
        min_length=min_length
    )
    
    payload = {
        "text": request.text,
        "params": {
            "max_length": request.max_length,
            "min_length": request.min_length,
            "num_beams": request.num_beams,
            "do_sample": request.do_sample,
            "early_stopping": request.early_stopping,
            "no_repeat_ngram_size": request.no_repeat_ngram_size,
            "temperature": request.temperature
        }
    }
    
    # Call SageMaker runtime
    try:
        runtime = boto3.client("sagemaker-runtime", region_name=region_name)
        logger.info(f"Sending request to endpoint {endpoint_name}")
        
        response = runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="application/json",
            Body=json.dumps(payload)
        )
        
        result = json.loads(response["Body"].read().decode())
        
        if "summary" in result:
            return result["summary"]
        else:
            logger.error(f"Unexpected response format: {result}")
            return "Error: Unable to generate summary"
    except Exception as e:
        logger.error(f"Error calling endpoint: {e}")
        raise RuntimeError(f"Failed to get summary from endpoint: {e}")
