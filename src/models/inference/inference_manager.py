"""Manages batch inference for transformer models."""

from typing import List, Optional, Any
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from .inference_dataclass import InferenceResult
from .inference_text_processor import process_input_texts, decode_outputs
from .model_runner import run_model_inference

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
            
            # Run inference
            inference_result = run_model_inference(
                self.model,
                processed,
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                length_penalty=length_penalty,
                early_stopping=early_stopping,
                extra_params=extra_params
            )
            
            # Decode outputs
            decoded_outputs = decode_outputs(
                inference_result.output_tokens, 
                self.tokenizer
            )
            
            # Create results
            return [
                InferenceResult(
                    output_text=output,
                    input_length=len(text.split()),
                    output_length=len(output.split()),
                    processing_duration_ms=inference_result.processing_duration_ms / len(texts),
                    input_token_count=len(self.tokenizer.encode(text))
                )
                for text, output in zip(texts, decoded_outputs)
            ]

        except Exception as e:
            raise RuntimeError(f"Batch inference failed: {str(e)}")
