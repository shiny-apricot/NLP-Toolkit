"""Text processing utilities for inference."""

from typing import List
import torch
from transformers import PreTrainedTokenizer
from .inference_dataclasses import ProcessingResult

def process_input_texts(
    texts: List[str],
    tokenizer: PreTrainedTokenizer,
    *,  # Force keyword arguments
    max_length: int,
    device: torch.device
) -> ProcessingResult:
    """Process and tokenize input texts."""
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    ).to(device)
    
    return ProcessingResult(
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"]
    )

def decode_outputs(
    outputs: torch.Tensor,
    tokenizer: PreTrainedTokenizer
) -> List[str]:
    """Decode model outputs to text."""
    return tokenizer.batch_decode(
        outputs,
        skip_special_tokens=True
    )
