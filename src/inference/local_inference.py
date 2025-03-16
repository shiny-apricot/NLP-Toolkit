"""
Module for running text summarization locally using saved models.
Supports both extractive and abstractive summarization approaches.
"""

import torch
from dataclasses import dataclass
from typing import Optional, Dict, Any, Union
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from ..utils.project_logger import get_logger


@dataclass
class LocalInferenceConfig:
    """Configuration for local inference."""
    model_path: Union[str, Path]
    device: str = "cpu"
    max_length: int = 150
    min_length: int = 50
    num_beams: int = 4
    early_stopping: bool = True


def summarize_text_locally(
    *,
    text: str,
    model_path: Union[str, Path],
    max_length: int = 150,
    min_length: int = 50,
    device: str = "cpu",
    logger: Optional[Any] = None
) -> str:
    """
    Generate summary of text using a local model.
    
    Args:
        text: Text to summarize
        model_path: Path to saved model
        max_length: Maximum length of the summary
        min_length: Minimum length of the summary
        device: Device to run inference on ('cpu' or 'cuda')
        logger: Optional logger instance
        
    Returns:
        Generated summary text
    """
    if logger is None:
        logger = get_logger("local_inference", level="INFO")
    
    # Convert model_path to Path object for consistency
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    # Set device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"
    
    # Load model and tokenizer
    logger.info(f"Loading model from {model_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        model = model.to(device)
        
        # Prepare input
        inputs = tokenizer(text, return_tensors="pt").to(device)
        
        # Generate summary
        logger.info("Generating summary")
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=max_length,
            min_length=min_length,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3
        )
        
        # Decode summary
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
        
    except Exception as e:
        logger.error(f"Error during local inference: {e}")
        raise RuntimeError(f"Failed to generate summary: {e}")
