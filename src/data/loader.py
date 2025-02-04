from typing import Optional, List, Union
from datasets import load_dataset, Dataset
import torch
from pathlib import Path
from transformers import PreTrainedTokenizer
from src.utils.project_logger import get_logger
from datetime import datetime
import json

class DatasetError(Exception):
    """Base exception for dataset-related errors."""
    pass

class HuggingFaceLoader:
    """Load and prepare datasets from HuggingFace hub."""
    
    def __init__(
        self,
        *,  # Force named parameters
        dataset_name: str,
        text_column: str = "text",
        summary_column: str = "summary",
        cache_dir: Optional[Path] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """Initialize the dataset loader."""
        self.dataset_name = dataset_name
        self.text_column = text_column
        self.summary_column = summary_column
        self.cache_dir = cache_dir
        self.device = device
        self.logger = get_logger(__name__)
        self._validate_params()

    def _validate_params(self) -> None:
        """Validate the initialization parameters."""
        if not self.dataset_name:
            raise DatasetError("Dataset name cannot be empty")
        
        self.logger.info(json.dumps({
            "event": "loader_initialized",
            "dataset": self.dataset_name,
            "device": self.device,
            "cache_dir": str(self.cache_dir)
        }))

    def load(
        self,
        *,  # Force named parameters
        split: str = "train",
        max_samples: Optional[int] = None,
        shuffle: bool = False
    ) -> Dataset:
        """Load dataset from HuggingFace hub."""
        start_time = datetime.now()
        
        try:
            self.logger.info(json.dumps({
                "event": "loading_dataset",
                "dataset": self.dataset_name,
                "split": split,
                "max_samples": max_samples
            }))

            dataset = load_dataset(
                self.dataset_name,
                split=split,
                cache_dir=str(self.cache_dir) if self.cache_dir else None
            )
            
            if shuffle:
                dataset = dataset.shuffle()
                
            if max_samples:
                dataset = dataset.select(range(min(max_samples, len(dataset))))

            required_columns = [self.text_column, self.summary_column]
            missing_columns = [col for col in required_columns if col not in dataset.column_names]
            
            if missing_columns:
                raise DatasetError(f"Missing required columns: {', '.join(missing_columns)}")

            processing_time = (datetime.now() - start_time).total_seconds()
            
            self.logger.info(json.dumps({
                "event": "dataset_loaded",
                "dataset": self.dataset_name,
                "samples": len(dataset),
                "processing_time": processing_time,
                "memory_usage": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            }))

            return dataset
            
        except Exception as e:
            self.logger.error(json.dumps({
                "event": "dataset_load_failed",
                "dataset": self.dataset_name,
                "error": str(e),
                "processing_time": (datetime.now() - start_time).total_seconds()
            }))
            raise DatasetError(f"Failed to load dataset {self.dataset_name}: {str(e)}") from e