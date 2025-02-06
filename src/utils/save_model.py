"""Model saving utility with rich metadata and logging."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any
import torch
import json
import shutil
from datetime import datetime
from transformers import PreTrainedModel, PreTrainedTokenizer

@dataclass
class ModelMetadata:
    """Essential model metadata."""
    model_name: str
    version: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    description: Optional[str] = None
    training_params: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)

def save_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    save_path: Path,
    *,  # Force keyword arguments
    version: str,
    description: Optional[str] = None,
    training_params: Optional[Dict[str, Any]] = None,
    performance_metrics: Optional[Dict[str, Any]] = None,
    compress: bool = True
) -> Path:
    """Save model with metadata."""
    save_path = Path(save_path)
    temp_path = save_path.parent / f"{save_path.name}.temp"

    try:
        # Save model and tokenizer
        model.save_pretrained(
            temp_path,
            save_function=torch.save if not compress else None
        )
        tokenizer.save_pretrained(temp_path)

        # Create and save metadata
        metadata = ModelMetadata(
            model_name=model.config.name_or_path,
            version=version,
            description=description,
            training_params=training_params or {},
            performance_metrics=performance_metrics or {}
        )

        with open(temp_path / "metadata.json", "w") as f:
            json.dump(metadata.__dict__, f, indent=2)

        # Safely move to final location
        if save_path.exists():
            shutil.rmtree(save_path)
        temp_path.rename(save_path)
        return save_path

    except Exception as e:
        if temp_path.exists():
            shutil.rmtree(temp_path)
        raise Exception(f"Failed to save model: {e}")