from dataclasses import dataclass
from typing import List, Optional, Dict, Union
import re
from transformers import PreTrainedTokenizer, AutoTokenizer
import torch
from logging import Logger
import nltk
from nltk.tokenize import sent_tokenize
import logging

@dataclass
class PreprocessingConfig:
    """Configuration for text preprocessing."""
    max_length: int
    min_length: int
    remove_urls: bool = True
    remove_html: bool = True
    normalize_whitespace: bool = True
    lowercase: bool = True
    model_name: str = "t5-base"

class TextPreprocessor:
    """Handle text preprocessing for summarization tasks."""
    
    def __init__(
        self,
        config: PreprocessingConfig,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        logger: Optional[Logger] = None
    ):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(config.model_name)
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

    def clean_text(self, text: str) -> str:
        """Clean raw text by removing unwanted elements."""
        if not isinstance(text, str):
            raise ValueError("Input must be a string")

        if self.config.remove_urls:
            text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        if self.config.remove_html:
            text = re.sub(r'<[^>]+>', '', text)
        
        if self.config.normalize_whitespace:
            text = ' '.join(text.split())
        
        if self.config.lowercase:
            text = text.lower()
            
        return text.strip()

    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using NLTK."""
        return sent_tokenize(text)

    def tokenize(
        self,
        text: str,
        return_tensors: str = "pt",
        truncation: bool = True,
        padding: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Tokenize text using the configured tokenizer."""
        try:
            return self.tokenizer(
                text,
                max_length=self.config.max_length,
                truncation=truncation,
                padding=padding,
                return_tensors=return_tensors
            )
        except Exception as e:
            self.logger.error(f"Tokenization failed: {str(e)}")
            raise

    def batch_process(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> List[Dict[str, torch.Tensor]]:
        """Process a batch of texts."""
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            cleaned_batch = [self.clean_text(text) for text in batch]
            
            try:
                tokens = self.tokenize(cleaned_batch)
                results.append(tokens)
            except Exception as e:
                self.logger.error(f"Batch processing failed at batch {i}: {str(e)}")
                raise
                
        return results

    def __call__(self, text: Union[str, List[str]]) -> Union[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]:
        """Process single text or list of texts."""
        if isinstance(text, str):
            cleaned = self.clean_text(text)
            return self.tokenize(cleaned)
        elif isinstance(text, list):
            return self.batch_process(text)
        else:
            raise ValueError("Input must be either string or list of strings")
