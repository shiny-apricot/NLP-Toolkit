"""Manages batch inference for transformer models."""

from typing import List, Optional, Any
import time
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from ...utils.gpu_manager import GPUManager, GPUConfig

from .inference_dataclasses import (
    ProcessingResult,
    InferenceStepResult,
    InferenceResult
)
from .processor import process_input_texts, decode_outputs

@dataclass
class InferenceResult:
    """Result from model inference."""
    output_text: str
    input_length: int
    output_length: int
    processing_duration_ms: float
    input_token_count: int

class InferenceManager:
    """Manages model inference with resource cleanup."""
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: str
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.gpu_config = GPUConfig(
            min_memory_mb=8 * 1024,
            max_memory_percent=0.9,
            strategy="data_parallel",
            prefer_bfloat16=True
        )

    def process_batch(
        self,
        texts: List[str],
        *,
        max_length: int,
        min_length: int,
        num_beams: int,
        length_penalty: float,
        early_stopping: bool
    ) -> List[InferenceResult]:
        results = []
        
        with GPUManager(self.gpu_config):
            for text in texts:
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                
                inputs = self.tokenizer(
                    text,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt"
                ).to(self.device)
                
                start_time.record()
                outputs = self.model.generate(
                    inputs["input_ids"],
                    max_length=max_length,
                    min_length=min_length,
                    num_beams=num_beams,
                    length_penalty=length_penalty,
                    early_stopping=early_stopping
                )
                end_time.record()
                torch.cuda.synchronize()
                
                summary = self.tokenizer.decode(
                    outputs[0],
                    skip_special_tokens=True
                )
                
                results.append(InferenceResult(
                    output_text=summary,
                    input_length=len(text.split()),
                    output_length=len(summary.split()),
                    processing_duration_ms=start_time.elapsed_time(end_time),
                    input_token_count=len(inputs["input_ids"][0])
                ))
                
        return results

    def __enter__(self):
        """Context manager entry."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
