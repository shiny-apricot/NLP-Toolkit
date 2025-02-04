from typing import List, Optional, Dict, Union
import re
from transformers import PreTrainedTokenizer, AutoTokenizer
import torch
from logging import Logger
import nltk
from nltk.tokenize import sent_tokenize
import logging

class TextPreprocessor:
    """Handle text preprocessing for summarization tasks."""
    
    def __init__(
        self,
        *,  # Force keyword arguments
        max_length: int,
        min_length: int,
        model_name: str = "t5-base",
        remove_urls: bool = True,
        remove_html: bool = True,
        normalize_whitespace: bool = True,
        lowercase: bool = True,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        logger: Optional[Logger] = None
    ):
        """
        Initialize text preprocessor with explicit parameters.
        
        Args:
            max_length: Maximum sequence length
            min_length: Minimum sequence length
            model_name: Name of the pretrained model for tokenizer
            remove_urls: Whether to remove URLs from text
            remove_html: Whether to remove HTML tags
            normalize_whitespace: Whether to normalize whitespace
            lowercase: Whether to convert text to lowercase
            tokenizer: Optional custom tokenizer
            logger: Optional custom logger
        """
        self.max_length = max_length
        self.min_length = min_length
        self.remove_urls = remove_urls
        self.remove_html = remove_html
        self.normalize_whitespace = normalize_whitespace
        self.lowercase = lowercase
        self.logger = logger or logging.getLogger(__name__)
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_name)
        
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

    def clean_text(self, text: str) -> str:
        """Clean raw text by removing unwanted elements."""
        if not isinstance(text, str):
            raise ValueError("Input must be a string")

        if self.remove_urls:
            text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        if self.remove_html:
            text = re.sub(r'<[^>]+>', '', text)
        
        if self.normalize_whitespace:
            text = ' '.join(text.split())
        
        if self.lowercase:
            text = text.lower()
            
        return text.strip()

    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using NLTK."""
        return sent_tokenize(text)

    def tokenize(
        self,
        text: str,
        *,  # Force keyword arguments
        return_tensors: str = "pt",
        truncation: bool = True,
        padding: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize text using the configured tokenizer.
        
        Args:
            text: Input text to tokenize
            return_tensors: Type of tensors to return
            truncation: Whether to truncate sequences
            padding: Whether to pad sequences
        """
        try:
            return self.tokenizer(
                text,
                max_length=self.max_length,
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
        *,  # Force keyword arguments
        batch_size: int = 32
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Process a batch of texts.
        
        Args:
            texts: List of input texts
            batch_size: Size of processing batches
        """
        if not texts:
            raise ValueError("Input texts list cannot be empty")
            
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
