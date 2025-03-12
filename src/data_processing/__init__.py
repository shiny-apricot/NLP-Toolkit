"""
Data processing module for text summarization.
Handles dataset loading, preprocessing, and batching operations.
"""

from .preprocessing import preprocess_text, ProcessingResult, TokenData
from .huggingface_loader import HuggingFaceLoader
from .dataset_types import ProcessedDataset, TokenizedBatch, DatasetError
from .dataset_config import DatasetConfig, DATASET_CONFIGS
from .batch_processor import process_dataset_in_batches

__all__ = [
    'preprocess_text',
    'ProcessingResult',
    'TokenData',
    'HuggingFaceLoader',
    'ProcessedDataset',
    'TokenizedBatch',
    'DatasetError',
    'DatasetConfig',
    'DATASET_CONFIGS',
    'process_dataset_in_batches'
]
