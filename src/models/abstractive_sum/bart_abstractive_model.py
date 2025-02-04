from dataclasses import dataclass
from typing import List, Optional, Union
import torch
from torch.utils.data import Dataset
from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
    Trainer,
    TrainingArguments
)
import logging
import json
from pathlib import Path

@dataclass
class SummarizationInput:
    text: Union[str, List[str]]
    max_length: int
    min_length: int

@dataclass
class SummarizationResult:
    summary: Union[str, List[str]]
    input_tokens: int
    output_tokens: int
    processing_time_ms: float

class SummarizationError(Exception):
    """Base class for summarization-specific errors."""
    pass

class ModelInitializationError(SummarizationError):
    """Raised when model initialization fails."""
    pass

class TokenLengthError(SummarizationError):
    """Raised when text exceeds model's maximum token length."""
    pass

@dataclass
class BartConfig:
    model_name: str = "facebook/bart-large-cnn"
    max_length: int = 1024
    min_length: int = 50
    length_penalty: float = 2.0
    num_beams: int = 4
    device: str = "auto"
    dtype: torch.dtype = torch.bfloat16

    def __post_init__(self):
        if self.max_length < self.min_length:
            raise ValueError("max_length must be greater than min_length")
        if self.length_penalty < 0:
            raise ValueError("length_penalty must be positive")
        if self.num_beams < 1:
            raise ValueError("num_beams must be at least 1")

class GPUManager:
    """Context manager for GPU memory management."""
    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

class BartSummarizer:
    """BART-based abstractive summarization model with multi-GPU support."""
    
    def __init__(self, config: BartConfig):
        self.config = config
        self._setup_logging()
        self.device_map = "auto" if torch.cuda.is_available() else None
        
        with GPUManager():
            self._initialize_model()

    def _setup_logging(self) -> None:
        """Initialize structured logging."""
        self.logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(json.dumps({
            "timestamp": "%(asctime)s",
            "level": "%(levelname)s",
            "message": "%(message)s"
        }))
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def _initialize_model(self) -> None:
        """Initialize BART model and tokenizer with proper configuration."""
        try:
            self.model = BartForConditionalGeneration.from_pretrained(
                self.config.model_name,
                device_map=self.device_map,
                torch_dtype=self.config.dtype
            )
            self.tokenizer = BartTokenizer.from_pretrained(self.config.model_name)
            
            self.logger.info(json.dumps({
                "event": "model_initialized",
                "model_name": self.config.model_name,
                "device": self.device_map,
                "dtype": str(self.config.dtype)
            }))
        except Exception as e:
            raise ModelInitializationError(f"Failed to initialize BART model: {e}")

    def summarize(self, text: Union[str, List[str]]) -> SummarizationResult:
        """
        Generate summary for input text(s).
        
        Args:
            text: Single string or list of strings to summarize
            
        Returns:
            SummarizationResult containing summary and metadata
            
        Raises:
            TokenLengthError: If input text exceeds model's maximum length
            SummarizationError: For other summarization-related errors
        """
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        try:
            with GPUManager():
                start_time.record()
                
                inputs = self.tokenizer(
                    text,
                    max_length=self.config.max_length,
                    truncation=True,
                    padding=True,
                    return_tensors="pt"
                )
                
                if torch.max(inputs["input_ids"]).item() >= self.tokenizer.model_max_length:
                    raise TokenLengthError("Input text exceeds maximum token length")
                
                if self.device_map is None:
                    inputs = inputs.to(self.model.device)
                
                summary_ids = self.model.generate(
                    inputs["input_ids"],
                    num_beams=self.config.num_beams,
                    min_length=self.config.min_length,
                    max_length=self.config.max_length,
                    length_penalty=self.config.length_penalty,
                    early_stopping=True
                )
                
                summaries = self.tokenizer.batch_decode(
                    summary_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
                
                end_time.record()
                torch.cuda.synchronize()
                
                result = SummarizationResult(
                    summary=summaries[0] if isinstance(text, str) else summaries,
                    input_tokens=inputs["input_ids"].shape[1],
                    output_tokens=summary_ids.shape[1],
                    processing_time_ms=start_time.elapsed_time(end_time)
                )
                
                self.logger.info(json.dumps({
                    "event": "summarization_complete",
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                    "processing_time_ms": result.processing_time_ms
                }))
                
                return result
                
        except Exception as e:
            self.logger.error(json.dumps({
                "event": "summarization_failed",
                "error": str(e)
            }))
            raise SummarizationError(f"Summarization failed: {e}")

    def train(
        self,
        dataset: Dataset,
        output_dir: Union[str, Path],
        training_args: Optional[TrainingArguments] = None
    ) -> None:
        """Fine-tune the model on a dataset."""
        if training_args is None:
            training_args = TrainingArguments(
                output_dir=str(output_dir),
                per_device_train_batch_size=4,
                gradient_accumulation_steps=4,
                learning_rate=3e-5,
                num_train_epochs=3,
                fp16=True if self.config.dtype == torch.float16 else False,
                bf16=True if self.config.dtype == torch.bfloat16 else False,
                save_strategy="epoch",
                logging_steps=100,
            )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=self.tokenizer,
        )
        
        try:
            trainer.train()
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise

    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> None:
        """Load model from checkpoint."""
        try:
            self.model = BartForConditionalGeneration.from_pretrained(
                checkpoint_path,
                device_map=self.device_map,
                torch_dtype=self.config.dtype
            )
            self.tokenizer = BartTokenizer.from_pretrained(checkpoint_path)
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            raise
