"""Text processing utilities for inference."""

from typing import List, Dict
import torch
from transformers import PreTrainedTokenizer
from .inference_dataclass import InferenceConfig

def process_input_texts(
    texts: List[str],
    tokenizer: PreTrainedTokenizer,
    config: InferenceConfig,
    device: torch.device
) -> Dict[str, torch.Tensor]:
    """Process and tokenize input texts."""
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=config.max_length,
        return_tensors="pt"
    ).to(device)

def decode_outputs(
    outputs: torch.Tensor,
    tokenizer: PreTrainedTokenizer
) -> List[str]:
    """Decode model outputs to text."""
    return tokenizer.batch_decode(
        outputs,
        skip_special_tokens=True
    )
