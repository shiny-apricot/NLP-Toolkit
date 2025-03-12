"""
BART-based abstractive summarization model with training and inference capabilities.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import torch
from transformers import BartForConditionalGeneration, BartTokenizer
import logging
from pathlib import Path
import time
from utils.save_model import save_model

@dataclass
class BartConfig:
    """Configuration for BART model."""
    model_name: str = "facebook/bart-large-cnn"
    max_length: int = 1024
    min_length: int = 50
    length_penalty: float = 2.0
    num_beams: int = 4
    device_map: str = "auto"
    use_bfloat16: bool = True

@dataclass
class SummarizationResult:
    """Structured output for summarization results."""
    summary: str
    input_tokens: int
    output_tokens: int
    processing_time: float
    attention_scores: Optional[Dict[str, float]] = None

class BartModelError(Exception):
    """Base class for BART model errors."""
    pass

class ModelInitializationError(BartModelError):
    """Raised when model initialization fails."""
    pass

class TokenizationError(BartModelError):
    """Raised when text tokenization fails."""
    pass

class BartModel:
    """Base class for BART summarization model."""
    
    def __init__(
        self,
        config: BartConfig,
        logger: Optional[logging.Logger] = None
    ):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        try:
            dtype = torch.bfloat16 if config.use_bfloat16 else torch.float32
            self.model = BartForConditionalGeneration.from_pretrained(
                config.model_name,
                device_map=config.device_map,
                torch_dtype=dtype
            )
            self.tokenizer = BartTokenizer.from_pretrained(config.model_name)
            
        except Exception as e:
            self.logger.error(f"Failed to initialize BART model: {e}")
            raise ModelInitializationError(f"Model initialization failed: {e}")

    def save(
        self,
        save_path: Path,
        *,
        version: str,
        description: Optional[str] = None,
        performance_metrics: Optional[dict] = None
    ) -> Path:
        """Save model with metadata."""
        training_params = {
            "max_length": self.config.max_length,
            "min_length": self.config.min_length,
            "length_penalty": self.config.length_penalty,
            "num_beams": self.config.num_beams
        }
        
        try:
            return save_model(
                model=self.model,
                tokenizer=self.tokenizer,
                save_path=save_path,
                version=version,
                description=description,
                training_params=training_params,
                performance_metrics=performance_metrics
            )
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
            raise

    def _tokenize(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenize input text."""
        try:
            return self.tokenizer(
                text,
                max_length=self.config.max_length,
                truncation=True,
                return_tensors="pt"
            ).to(self.model.device)
        except Exception as e:
            raise TokenizationError(f"Failed to tokenize text: {e}")

    def _decode(self, token_ids: torch.Tensor) -> str:
        """Decode token IDs to text."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
        
    def summarize(
        self,
        text: str,
        *,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None
    ) -> SummarizationResult:
        """Generate a summary for the given text.
        
        Args:
            text: Input text to summarize
            max_length: Maximum length of summary (overrides config)
            min_length: Minimum length of summary (overrides config)
            
        Returns:
            SummarizationResult containing the summary and metadata
        """
        start_time = time.time()
        
        # Use provided values or fall back to config
        max_len = max_length or self.config.max_length
        min_len = min_length or self.config.min_length
        
        try:
            # Tokenize input
            inputs = self._tokenize(text)
            input_tokens = inputs["input_ids"].shape[1]
            
            # Generate summary
            summary_ids = self.model.generate(
                inputs["input_ids"],
                max_length=max_len,
                min_length=min_len,
                length_penalty=self.config.length_penalty,
                num_beams=self.config.num_beams,
                early_stopping=True
            )
            
            # Decode the summary
            summary = self._decode(summary_ids[0])
            processing_time = time.time() - start_time
            
            return SummarizationResult(
                summary=summary,
                input_tokens=input_tokens,
                output_tokens=len(summary_ids[0]),
                processing_time=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"Summarization failed: {e}")
            raise
