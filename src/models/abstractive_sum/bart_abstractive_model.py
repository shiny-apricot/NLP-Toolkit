"""
BART-based abstractive summarization model implementation.
Uses pretrained BART model for text summarization with optimized inference.
"""

from dataclasses import dataclass
from typing import List, Optional
import torch 
from transformers import BartForConditionalGeneration, BartTokenizer
import logging
from pathlib import Path
from utils.save_model import save_model


class ModelInitializationError(Exception):
    """Raised when model initialization fails."""
    pass

class TokenizationError(Exception):
    """Raised when text tokenization fails."""
    pass

@dataclass 
class SummarizationResult:
    """Structured output for summarization results."""
    summary: str
    input_tokens: int
    output_tokens: int
    processing_time: float

class BartAbstractiveSummarizer:
    """BART-based abstractive summarization model."""
    
    def __init__(
        self,
        model_name: str = "facebook/bart-large-cnn",
        *,  # Force named parameters
        max_length: int = 1024,
        min_length: int = 50,
        length_penalty: float = 2.0, 
        num_beams: int = 4,
        device_map: str = "auto",
        use_bfloat16: bool = True,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize BART summarizer.

        Args:
            model_name: Pretrained model name/path
            max_length: Maximum output length 
            min_length: Minimum output length
            length_penalty: Length penalty factor
            num_beams: Number of beams for beam search
            device_map: Device mapping strategy
            use_bfloat16: Whether to use bfloat16
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        
        try:
            dtype = torch.bfloat16 if use_bfloat16 else torch.float32
            self.model = BartForConditionalGeneration.from_pretrained(
                model_name,
                device_map=device_map,
                torch_dtype=dtype
            )
            self.tokenizer = BartTokenizer.from_pretrained(model_name)
            self.max_length = max_length
            self.min_length = min_length
            self.length_penalty = length_penalty
            self.num_beams = num_beams
            
        except Exception as e:
            self.logger.error(f"Failed to initialize BART model: {e}")
            raise ModelInitializationError(f"Model initialization failed: {e}")

    def save(
        self,
        save_path: Path,
        *,  # Force named parameters
        version: str,
        description: Optional[str] = None,
        performance_metrics: Optional[dict] = None,
        s3_bucket: Optional[str] = None,
        aws_region: Optional[str] = None
    ) -> Path:
        """
        Save model with metadata.

        Args:
            save_path: Path to save model
            version: Model version
            description: Optional model description
            performance_metrics: Optional performance metrics
            s3_bucket: Optional S3 bucket for upload
            aws_region: Optional AWS region
        """
        
        training_params = {
            "max_length": self.max_length,
            "min_length": self.min_length,
            "length_penalty": self.length_penalty,
            "num_beams": self.num_beams
        }
        
        try:
            return save_model(
                model=self.model,
                tokenizer=self.tokenizer,
                save_path=save_path,
                version=version,
                description=description,
                training_params=training_params,
                performance_metrics=performance_metrics,
                s3_bucket=s3_bucket,
                aws_region=aws_region
            )
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
            raise

    def summarize(
        self,
        text: str,
        *,  # Force named parameters
        max_length: Optional[int] = None,
        min_length: Optional[int] = None
    ) -> SummarizationResult:
        """
        Generate abstractive summary of input text.

        Args:
            text: Input text to summarize
            max_length: Optional override for max length
            min_length: Optional override for min length

        Returns:
            SummarizationResult containing summary and metadata

        Raises:
            TokenizationError: If tokenization fails
        """
        max_length = max_length or self.max_length
        min_length = min_length or self.min_length

        try:
            with GPUManager():
                inputs = self.tokenizer(
                    text,
                    max_length=self.max_length,
                    truncation=True,
                    return_tensors="pt"
                ).to(self.model.device)

                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                
                start_time.record()
                outputs = self.model.generate(
                    inputs["input_ids"],
                    max_length=max_length,
                    min_length=min_length, 
                    length_penalty=self.length_penalty,
                    num_beams=self.num_beams,
                    early_stopping=True
                )
                end_time.record()
                torch.cuda.synchronize()
                
                summary = self.tokenizer.decode(
                    outputs[0],
                    skip_special_tokens=True
                )
                
                return SummarizationResult(
                    summary=summary,
                    input_tokens=len(inputs["input_ids"][0]),
                    output_tokens=len(outputs[0]),
                    processing_time=start_time.elapsed_time(end_time) / 1000
                )

        except Exception as e:
            self.logger.error(f"Summarization failed: {e}")
            raise TokenizationError(f"Failed to process text: {e}")

class GPUManager:
    """Context manager for GPU memory management."""
    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
