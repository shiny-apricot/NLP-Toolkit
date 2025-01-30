from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
    Trainer,
    TrainingArguments
)
import logging
from pathlib import Path
import numpy as np

@dataclass
class BartConfig:
    model_name: str = "facebook/bart-large-cnn"
    max_length: int = 1024
    min_length: int = 50
    length_penalty: float = 2.0
    num_beams: int = 4
    device: str = "auto"
    dtype: torch.dtype = torch.bfloat16

class BartSummarizer:
    """BART-based abstractive summarization model with multi-GPU support."""
    
    def __init__(self, config: BartConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.device_map = "auto" if torch.cuda.is_available() else None
        
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Initialize BART model and tokenizer with proper configuration."""
        try:
            self.model = BartForConditionalGeneration.from_pretrained(
                self.config.model_name,
                device_map=self.device_map,
                torch_dtype=self.config.dtype
            )
            self.tokenizer = BartTokenizer.from_pretrained(self.config.model_name)
        except Exception as e:
            self.logger.error(f"Failed to initialize BART model: {e}")
            raise
    
    def summarize(self, text: Union[str, List[str]]) -> Union[str, List[str]]:
        """Generate summary for input text(s)."""
        try:
            inputs = self.tokenizer(
                text,
                max_length=self.config.max_length,
                truncation=True,
                padding=True,
                return_tensors="pt"
            )
            
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
                summary_ids, skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            return summaries[0] if isinstance(text, str) else summaries
            
        except Exception as e:
            self.logger.error(f"Summarization failed: {e}")
            raise
    
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
