from dataclasses import dataclass
from typing import Optional, List, Union, Literal
from datasets import load_dataset, Dataset
import torch
from pathlib import Path
from transformers import PreTrainedTokenizer, AutoTokenizer
from src.utils.project_logger import get_logger
from preprocessing import TokenData, preprocess_text, ProcessingResult
from datetime import datetime
import json

@dataclass
class DatasetConfig:
    """Configuration for dataset loading."""
    dataset_name: str
    version: str
    text_column: str
    summary_column: str

@dataclass
class PreprocessingParams:
    """Parameters for text preprocessing."""
    remove_urls: bool = False
    remove_html: bool = False
    normalize_whitespace: bool = True
    lowercase: bool = False

@dataclass
class TokenBatch:
    """Batch of tokenized text."""
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    token_type_ids: Optional[torch.Tensor] = None

@dataclass
class TokenizedDataset:
    """Structured container for processed dataset results."""
    dataset: Dataset
    text_tokens: TokenBatch
    summary_tokens: TokenBatch
    raw_texts: List[str]
    raw_summaries: List[str]
    processing_time: float

class DatasetError(Exception):
    """Base exception for dataset-related errors."""
    pass

class HuggingFaceLoader:
    """Load and prepare datasets from HuggingFace hub."""
    
    CNN_DAILY_CONFIG = DatasetConfig(
        dataset_name="cnn_dailymail",
        version="3.0.0",
        text_column="article",
        summary_column="highlights"
    )
    
    def __init__(
        self,
        *,  # Force named parameters
        dataset_name: str,
        text_column: str = "text",
        summary_column: str = "summary",
        cache_dir: Optional[Path] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        tokenizer: Optional[PreTrainedTokenizer] = None,
        max_length: int = 512,
        summary_max_length: int = 128
    ):
        """Initialize the dataset loader."""
        self.dataset_name = dataset_name
        self.text_column = text_column
        self.summary_column = summary_column
        self.cache_dir = cache_dir
        self.device = device
        self.logger = get_logger(__name__)
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
        self.max_length = max_length
        self.summary_max_length = summary_max_length
        self._validate_params()

    @classmethod
    def load_cnn_daily(
        cls,
        *,
        split: Literal["train", "validation", "test"] = "train",
        max_samples: Optional[int] = None,
        shuffle: bool = False,
        cache_dir: Optional[Path] = None
    ) -> Dataset:
        """
        Load CNN/DailyMail dataset with optimized settings.
        
        Args:
            split: Dataset split to load
            max_samples: Maximum number of samples to load
            shuffle: Whether to shuffle the dataset
            cache_dir: Directory to cache the dataset
            
        Returns:
            Dataset: Loaded CNN/DailyMail dataset
        """
        loader = cls(
            dataset_name=f"{cls.CNN_DAILY_CONFIG.dataset_name}/{cls.CNN_DAILY_CONFIG.version}",
            text_column=cls.CNN_DAILY_CONFIG.text_column,
            summary_column=cls.CNN_DAILY_CONFIG.summary_column,
            cache_dir=cache_dir
        )
        
        return loader.load(
            split=split,
            max_samples=max_samples,
            shuffle=shuffle
        )

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

    def load_and_preprocess(
        self,
        *,
        split: str = "train",
        max_samples: Optional[int] = None,
        shuffle: bool = False,
        preprocessing_params: Optional[PreprocessingParams] = None
    ) -> TokenizedDataset:
        """Load and preprocess dataset with tokenization."""
        start_time = datetime.now()
        
        dataset = self.load(split=split, max_samples=max_samples, shuffle=shuffle)
        params = preprocessing_params or PreprocessingParams()
        
        processed_texts = [
            preprocess_text(
                text=text,
                max_length=self.max_length,
                min_length=0,
                tokenizer=self.tokenizer,
                remove_urls=params.remove_urls,
                remove_html=params.remove_html,
                normalize_whitespace=params.normalize_whitespace,
                lowercase=params.lowercase,
                logger=self.logger
            )
            for text in dataset[self.text_column]
        ]
        
        processed_summaries = [
            preprocess_text(
                text=summary,
                max_length=self.summary_max_length,
                min_length=0,
                tokenizer=self.tokenizer,
                remove_urls=params.remove_urls,
                remove_html=params.remove_html,
                normalize_whitespace=params.normalize_whitespace,
                lowercase=params.lowercase,
                logger=self.logger
            )
            for summary in dataset[self.summary_column]
        ]
        
        text_tokens = self._create_token_batch(
            [p.tokens for p in processed_texts]
        )
        
        summary_tokens = self._create_token_batch(
            [p.tokens for p in processed_summaries]
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        self.logger.info(json.dumps({
            "event": "dataset_processed",
            "samples": len(dataset),
            "processing_time": processing_time
        }))
        
        return TokenizedDataset(
            dataset=dataset,
            text_tokens=text_tokens,
            summary_tokens=summary_tokens,
            raw_texts=[p.cleaned_text for p in processed_texts],
            raw_summaries=[p.cleaned_text for p in processed_summaries],
            processing_time=processing_time
        )

    def _create_token_batch(
        self,
        token_dicts: List[TokenData]
    ) -> TokenBatch:
        """Convert list of token dictionaries to TokenBatch."""
        padded = self.tokenizer.pad(token_dicts, padding=True, return_tensors="pt")
        return TokenBatch(
            input_ids=padded["input_ids"],
            attention_mask=padded["attention_mask"],
            token_type_ids=padded.get("token_type_ids")
        )