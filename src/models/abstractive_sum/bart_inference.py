"""BART model inference functionality."""

from typing import List, Optional, Dict
import torch
from dataclasses import dataclass
from .bart_abstractive_model import (
    BartBaseModel,
    BartConfig,
    SummarizationResult
)


@dataclass
class InferenceConfig:
    """Inference-specific configuration."""
    batch_size: int = 8
    max_length: Optional[int] = None
    min_length: Optional[int] = None
    num_return_sequences: int = 1
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 3
    early_stopping: bool = True


class BartInference(BartBaseModel):
    """BART model inference implementation."""
    
    def __init__(self, model_config: BartConfig, inference_config: InferenceConfig):
        super().__init__(model_config)
        self.inference_config = inference_config

    def summarize(
        self,
        text: str,
        *,
        return_attention: bool = False
    ) -> SummarizationResult:
        """Generate summary for single text."""
        with torch.cuda.amp.autocast(enabled=self.config.use_bfloat16):
            with torch.no_grad():
                inputs = self._tokenize(text)
                
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                
                start.record()
                outputs = self.model.generate(
                    inputs["input_ids"],
                    max_length=self.inference_config.max_length or self.config.max_length,
                    min_length=self.inference_config.min_length or self.config.min_length,
                    length_penalty=self.config.length_penalty,
                    num_beams=self.config.num_beams,
                    early_stopping=self.inference_config.early_stopping,
                    num_return_sequences=self.inference_config.num_return_sequences,
                    repetition_penalty=self.inference_config.repetition_penalty,
                    no_repeat_ngram_size=self.inference_config.no_repeat_ngram_size,
                    output_attentions=return_attention,
                    return_dict_in_generate=True
                )
                end.record()
                
                torch.cuda.synchronize()
                
                summary = self._decode(outputs.sequences[0])
                attention_scores = self._process_attention(outputs) if return_attention else None
                
                return SummarizationResult(
                    summary=summary,
                    input_tokens=len(inputs["input_ids"][0]),
                    output_tokens=len(outputs.sequences[0]),
                    processing_time=start.elapsed_time(end) / 1000,
                    attention_scores=attention_scores
                )

    def summarize_batch(
        self,
        texts: List[str],
        **kwargs
    ) -> List[SummarizationResult]:
        """Generate summaries for text batch."""
        results = []
        for i in range(0, len(texts), self.inference_config.batch_size):
            batch = texts[i:i + self.inference_config.batch_size]
            results.extend([self.summarize(text, **kwargs) for text in batch])
        return results

    def _process_attention(self, outputs) -> Optional[Dict[str, float]]:
        """Process attention weights from model output."""
        if not hasattr(outputs, "attentions"):
            return None
            
        # Average attention scores across layers and heads
        attention = torch.stack(outputs.attentions).mean(dim=[0, 1])  # [batch, seq, seq]
        return {
            "mean_attention": attention.mean().item(),
            "max_attention": attention.max().item()
        }

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
