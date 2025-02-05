"""Type definitions for inference components."""

from dataclasses import dataclass
from typing import List
import torch

@dataclass
class ProcessingResult:
    """Result of text processing step."""
    input_ids: torch.Tensor
    attention_mask: torch.Tensor

@dataclass
class InferenceStepResult:
    """Result of model inference step."""
    output_tokens: torch.Tensor
    processing_duration_ms: float

@dataclass
class InferenceResult:
    """Final result of inference pipeline."""
    output_text: str
    input_length: int
    output_length: int
    processing_duration_ms: float
    input_token_count: int
