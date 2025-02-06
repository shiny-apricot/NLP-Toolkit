from dataclasses import dataclass
from typing import List, Optional, Dict, Union
import re
from transformers import PreTrainedTokenizer, AutoTokenizer
import torch
import logging
from nltk.tokenize import sent_tokenize

@dataclass
class TokenData:
    """Container for tokenized text data."""
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    token_type_ids: Optional[torch.Tensor] = None

@dataclass
class ProcessingResult:
    cleaned_text: str
    sentences: List[str]
    tokens: Optional[TokenData] = None

def preprocess_text(
    text: str,
    *,  # Force keyword arguments
    max_length: int,
    min_length: int,
    tokenizer: PreTrainedTokenizer,
    remove_urls: bool = False,
    remove_html: bool = False,
    normalize_whitespace: bool = True,
    lowercase: bool = False,
    logger: Optional[logging.Logger] = None
) -> ProcessingResult:
    """Process input text for summarization."""
    logger = logger or logging.getLogger(__name__)

    # Clean text
    cleaned = clean_text(
        text,
        remove_urls=remove_urls,
        remove_html=remove_html,
        normalize_whitespace=normalize_whitespace,
        lowercase=lowercase
    )
    
    # Split into sentences
    sentences = sent_tokenize(cleaned)
    
    # Tokenize
    tokens = tokenize_text(
        cleaned,
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    return ProcessingResult(
        cleaned_text=cleaned,
        sentences=sentences,
        tokens=tokens
    )

def clean_text(
    text: str,
    *,
    remove_urls: bool,
    remove_html: bool,
    normalize_whitespace: bool,
    lowercase: bool
) -> str:
    """Clean raw text by removing unwanted elements."""
    if not isinstance(text, str):
        raise ValueError("Input must be a string")

    if remove_urls:
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    if remove_html:
        text = re.sub(r'<[^>]+>', '', text)
    
    if normalize_whitespace:
        text = ' '.join(text.split())
    
    if lowercase:
        text = text.lower()
        
    return text.strip()

def tokenize_text(
    text: str,
    *,
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    return_tensors: str = "pt",
    truncation: bool = True,
    padding: bool = True
) -> TokenData:
    """Tokenize text using the provided tokenizer."""
    tokens = tokenizer(
        text,
        max_length=max_length,
        truncation=truncation,
        padding=padding,
        return_tensors=return_tensors
    )
    
    return TokenData(
        input_ids=tokens["input_ids"],
        attention_mask=tokens["attention_mask"],
        token_type_ids=tokens.get("token_type_ids")
    )
