"""Model inference operations."""

import time
import torch
from transformers import PreTrainedModel
from typing import Optional, Any
from .inference_dataclasses import ProcessingResult, InferenceStepResult

def run_model_inference(
    model: PreTrainedModel,
    processed_input: ProcessingResult,
    *,
    max_length: int,
    min_length: Optional[int] = None,
    num_beams: int = 1,
    length_penalty: float = 1.0,
    early_stopping: bool = False,
    extra_params: dict[str, Any] = None
) -> InferenceStepResult:
    """Run model inference and return outputs with timing."""
    with torch.no_grad():
        start_time = time.perf_counter()
        outputs = model.generate(
            processed_input.input_ids,
            attention_mask=processed_input.attention_mask,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            length_penalty=length_penalty,
            early_stopping=early_stopping,
            **(extra_params or {})
        )
        proc_time = (time.perf_counter() - start_time) * 1000
        
    return InferenceStepResult(
        output_tokens=outputs,
        processing_duration_ms=proc_time
    )
