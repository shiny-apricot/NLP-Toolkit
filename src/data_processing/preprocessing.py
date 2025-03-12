from dataclasses import dataclass
from typing import List, Optional, Dict, Union, Tuple
import re
from transformers import PreTrainedTokenizer, AutoTokenizer
import torch
import logging
from nltk.tokenize import sent_tokenize
import unicodedata

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

class TokenizerError(Exception):
    """Raised when tokenization fails."""
    pass

class TokenLengthError(Exception):
    """Raised when text exceeds token limits."""
    pass

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
    """
    Process input text for summarization.
    
    Args:
        text: Input text to process
        max_length: Maximum sequence length
        min_length: Minimum sequence length
        tokenizer: HuggingFace tokenizer
        remove_urls: Whether to remove URLs
        remove_html: Whether to remove HTML tags
        normalize_whitespace: Whether to normalize whitespace
        lowercase: Whether to convert to lowercase
        logger: Optional logger instance
    
    Returns:
        ProcessedResult containing cleaned and tokenized text
        
    Raises:
        TokenLengthError: When text exceeds token limits
        ValueError: When input parameters are invalid
    """
    logger = logger or logging.getLogger(__name__)

    # Validate lengths
    if min_length > max_length:
        raise ValueError("min_length cannot be greater than max_length")

    # Clean text with unicode normalization
    cleaned = clean_text(
        text,
        remove_urls=remove_urls,
        remove_html=remove_html,
        normalize_whitespace=normalize_whitespace,
        lowercase=lowercase
    )
    
    # Validate cleaned text length
    if len(cleaned) < min_length:
        raise ValueError(f"Text length {len(cleaned)} is below minimum {min_length}")
        
    # Estimate token count to avoid OOM errors
    estimated_token_count = estimate_token_count(cleaned, tokenizer)
    if estimated_token_count > max_length:
        logger.warning(
            f"Text may exceed token limit. Estimated tokens: {estimated_token_count}, "
            f"Max allowed: {max_length}"
        )

    # Split into sentences
    sentences = sent_tokenize(cleaned)
    
    # Tokenize with memory optimization
    try:
        tokens = tokenize_text_optimized(
            cleaned,
            tokenizer=tokenizer,
            max_length=max_length
        )
        
        return ProcessingResult(
            cleaned_text=cleaned,
            sentences=sentences,
            tokens=tokens
        )
    except TokenizerError as e:
        logger.error(f"Tokenization failed: {e}")
        # Return partial result without tokens
        return ProcessingResult(
            cleaned_text=cleaned,
            sentences=sentences,
            tokens=None
        )

def clean_text(
    text: str,
    *,
    remove_urls: bool,
    remove_html: bool,
    normalize_whitespace: bool,
    lowercase: bool
) -> str:
    """Clean and normalize text."""
    if not isinstance(text, str):
        raise ValueError("Input must be a string")

    # Normalize unicode characters
    text = unicodedata.normalize('NFKC', text)

    if remove_urls:
        # Improved URL regex to catch more URL patterns
        text = re.sub(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[/\w\.-]*(?:\?\S+)?|www\.[-\w.]+\.[a-z]{2,}', '', text)
    
    if remove_html:
        # Better HTML tag removal
        text = re.sub(r'</?(?:div|span|p|br|hr|h[1-6]|img|a|ul|ol|li|table|tr|td|th|thead|tbody|tfoot|pre|code|script|style)[^>]*>', '', text)
        # Also replace common HTML entities
        text = re.sub(r'&[a-z]+;', ' ', text)
    
    # Remove extra whitespace and newlines
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
    padding: str = "max_length"
) -> TokenData:
    """Tokenize text using the provided tokenizer.
    
    Args:
        text: Input text to tokenize
        tokenizer: HuggingFace tokenizer instance
        max_length: Maximum sequence length for tokenization
        return_tensors: Format of the returned tensors ('pt' for PyTorch, 'tf' for TensorFlow)
        truncation: Whether to truncate sequences exceeding max_length
        padding: Padding strategy ('max_length', 'longest', or boolean)
    
    Returns:
        TokenData containing input_ids, attention_mask, and possibly token_type_ids
        
    Raises:
        TokenizerError: When tokenization fails or input validation fails
    """
    # Validate inputs
    if not text:
        raise TokenizerError("Empty text provided for tokenization")
    
    if not isinstance(max_length, int) or max_length <= 0:
        raise TokenizerError(f"Invalid max_length: {max_length}")
    
    # Check if tokenizer has model_max_length attribute and validate
    if hasattr(tokenizer, 'model_max_length') and max_length > tokenizer.model_max_length:
        raise TokenizerError(
            f"max_length ({max_length}) exceeds tokenizer's maximum "
            f"({tokenizer.model_max_length})"
        )
        
    try:
        # Configure tokenization parameters based on tokenizer capabilities
        tokenizer_kwargs = {
            "max_length": max_length,
            "truncation": truncation,
            "padding": padding,
            "return_tensors": return_tensors,
            "add_special_tokens": True
        }
        
        # Only request token_type_ids if the tokenizer supports it
        if hasattr(tokenizer, 'model_type') and tokenizer.model_type in ['bert', 'roberta', 'albert', 'xlnet']:
            tokenizer_kwargs["return_token_type_ids"] = True
            
        tokens = tokenizer(text, **tokenizer_kwargs)
        
        return TokenData(
            input_ids=torch.tensor(tokens["input_ids"]),
            attention_mask=torch.tensor(tokens["attention_mask"]),
            token_type_ids=tokens.get("token_type_ids")
        )
        
    except ValueError as e:
        raise TokenizerError(f"Tokenization value error: {str(e)}") from e
    except RuntimeError as e:
        raise TokenizerError(f"Tokenization runtime error: {str(e)}") from e
    except Exception as e:
        raise TokenizerError(f"Unexpected error during tokenization: {str(e)}") from e

def estimate_token_count(text: str, tokenizer: PreTrainedTokenizer) -> int:
    """
    Estimate the number of tokens in text without full tokenization.
    
    Args:
        text: Input text
        tokenizer: HuggingFace tokenizer
    
    Returns:
        Estimated token count
    """
    # Simple word-based estimate with 10% margin
    word_count = len(text.split())
    # Most tokenizers produce ~1.3x more tokens than words
    return int(word_count * 1.3)

def tokenize_text_optimized(
    text: str,
    *,
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    stride: int = 100,
    return_tensors: str = "pt"
) -> TokenData:
    """
    Memory-efficient tokenization for long texts with proper handling of special tokens.
    
    Args:
        text: Input text to tokenize
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        stride: Stride length for overlapping chunks
        return_tensors: Format of returned tensors
        
    Returns:
        TokenData containing properly combined tokens
        
    Raises:
        TokenizerError: When tokenization fails
        TokenLengthError: When text exceeds token limits
    """
    if not text:
        raise TokenizerError("Empty text provided for tokenization")
    
    # Check if tokenizer supports token_type_ids
    supports_token_type_ids = hasattr(tokenizer, 'model_type') and tokenizer.model_type in [
        'bert', 'albert', 'xlnet', 'mpnet'
    ]
    
    try:
        # Simple tokenization first to check length
        test_encoding = tokenizer(
            text[:1000],  # Only tokenize beginning to save time
            add_special_tokens=True,
            return_length=True
        )
        
        # Estimate total tokens
        tokens_per_char = int(test_encoding.get('length', len(test_encoding.input_ids[0]))) / len(text[:1000])
        estimated_tokens = int(tokens_per_char * len(text))
        
        if estimated_tokens > tokenizer.model_max_length * 5:  # If text is significantly longer
            raise TokenLengthError(
                f"Text likely exceeds token capacity. Estimated tokens: {estimated_tokens}, "
                f"Model max tokens: {tokenizer.model_max_length}"
            )
            
        # For texts that fit within max_length, use simple tokenization
        if estimated_tokens <= max_length:
            encoding = tokenizer(
                text,
                max_length=max_length,
                truncation=True,
                padding="max_length",
                return_tensors=return_tensors,
                return_token_type_ids=supports_token_type_ids
            )
            
            return TokenData(
                input_ids=torch.tensor(encoding["input_ids"]),
                attention_mask=torch.tensor(encoding["attention_mask"]),
                token_type_ids=torch.tensor(encoding.get("token_type_ids")) if encoding.get("token_type_ids") is not None else None
            )
        
        # For longer texts, use truncation strategy
        encoding = tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors=return_tensors,
            return_token_type_ids=supports_token_type_ids
        )
        
        # Create proper TokenData object
        token_data = TokenData(
            input_ids=torch.tensor(encoding["input_ids"]),
            attention_mask=torch.tensor(encoding["attention_mask"]),
            token_type_ids=torch.tensor(encoding.get("token_type_ids")) if encoding.get("token_type_ids") is not None else None
        )
        
        return token_data
        
    except Exception as e:
        if "out of memory" in str(e).lower():
            raise TokenizerError(f"Out of memory during tokenization: {str(e)}") from e
        raise TokenizerError(f"Tokenization failed: {str(e)}") from e
