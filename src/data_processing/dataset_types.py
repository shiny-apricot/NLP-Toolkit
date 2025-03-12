"""Data container classes for dataset processing."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import torch
from datasets import Dataset


@dataclass
class TokenizedBatch:
    """Memory-efficient batch container."""
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: Optional[torch.Tensor] = None
    token_type_ids: Optional[torch.Tensor] = None
    
    def to(self, device: torch.device) -> 'TokenizedBatch':
        """Move batch to specified device."""
        return TokenizedBatch(
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            labels=self.labels.to(device) if self.labels is not None else None,
            token_type_ids=self.token_type_ids.to(device) if self.token_type_ids is not None else None
        )


@dataclass
class ProcessedDataset:
    """Container for processed dataset with metadata."""
    dataset: Dataset
    num_samples: int
    processing_time: float
    raw_texts: List[str]
    raw_summaries: List[str]
    cache_hit: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class DatasetError(Exception):
    """Base exception for dataset-related errors."""
    pass
