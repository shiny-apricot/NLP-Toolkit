"""Manages batch inference for transformer models."""

from typing import List, Optional, Any
import time
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from .inference_dataclasses import (
    ProcessingResult,
    InferenceStepResult,
    InferenceResult
)
from .processor import process_input_texts, decode_outputs

class InferenceManager:
    """Handles batch inference for transformer models."""
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: torch.device
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()  # Ensure model is in evaluation mode

    def process_batch(
        self,
        texts: List[str],
        *,  # Force keyword arguments
        max_length: int,
        min_length: Optional[int] = None,
        num_beams: int = 1,
        length_penalty: float = 1.0,
        early_stopping: bool = False,
        extra_params: dict[str, Any] = None
    ) -> List[InferenceResult]:
        """Process a batch of texts through the model."""
        try:
            # Process inputs
            processed = process_input_texts(
                texts, 
                self.tokenizer,
                max_length=max_length,
                device=self.device
            )
            
            # Run model inference
            with torch.no_grad():
                start_time = time.perf_counter()
                
                outputs = self.model.generate(
                    processed.input_ids,
                    attention_mask=processed.attention_mask,
                    max_length=max_length,
                    min_length=min_length,
                    num_beams=num_beams,
                    length_penalty=length_penalty,
                    early_stopping=early_stopping,
                    **(extra_params or {})
                )
                
                processing_time = (time.perf_counter() - start_time) * 1000  # Convert to ms
            
            # Decode outputs
            decoded_outputs = decode_outputs(outputs, self.tokenizer)
            
            # Create results
            return [
                InferenceResult(
                    output_text=output,
                    input_length=len(text.split()),
                    output_length=len(output.split()),
                    processing_duration_ms=processing_time / len(texts),
                    input_token_count=len(self.tokenizer.encode(text))
                )
                for text, output in zip(texts, decoded_outputs)
            ]

        except Exception as e:
            raise RuntimeError(f"Batch inference failed: {str(e)}") from e

    def __enter__(self):
        """Context manager entry."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
