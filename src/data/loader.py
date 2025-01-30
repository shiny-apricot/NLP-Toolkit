from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Union
import boto3
from pathlib import Path
import logging
from datasets import load_dataset, Dataset, DatasetDict
import json
import torch
from transformers import PreTrainedTokenizer
from .preprocessing import TextPreprocessor, PreprocessingConfig
from config_loader import InstanceConfig


@dataclass
class DatasetConfig:
    """Simplified configuration for dataset loading."""
    dataset_name: str
    text_column: str = "text"
    summary_column: str = "summary"
    split: str = "train"
    max_samples: Optional[int] = None
    cache_dir: Optional[str] = None

class HuggingFaceLoader:
    """Simplified loader focused on HuggingFace datasets."""
    
    def __init__(
        self,
        dataset_name: str,
        instance_config: InstanceConfig,
        text_column: str = "text",
        summary_column: str = "summary"
    ):
        self.config = DatasetConfig(
            dataset_name=dataset_name,
            text_column=text_column,
            summary_column=summary_column,
            cache_dir=instance_config.cache_dir
        )
        self.instance_config = instance_config
        
    def load(self) -> Dataset:
        """Load and prepare dataset in one step."""
        dataset = load_dataset(
            self.config.dataset_name,
            split=self.config.split,
            cache_dir=self.config.cache_dir
        )
        
        if self.config.max_samples:
            dataset = dataset.select(range(self.config.max_samples))
            
        return dataset