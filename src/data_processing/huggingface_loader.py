"""Main HuggingFace dataset loader class."""

from typing import Optional, Literal, Dict, Union
import torch
from pathlib import Path
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast, AutoTokenizer
from datasets import load_dataset
from datetime import datetime

from ..utils.project_logger import get_logger
from .dataset_config import DatasetConfig, DATASET_CONFIGS
from .dataset_types import ProcessedDataset, DatasetError
from .batch_processor import process_dataset_in_batches

def validate_tokenizer(tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]) -> None:
    """Validate tokenizer configuration."""
    if not hasattr(tokenizer, 'pad_token'):
        tokenizer.pad_token = tokenizer.eos_token
    if not hasattr(tokenizer, 'model_max_length'):
        tokenizer.model_max_length = 1024  # Default max length

class HuggingFaceLoader:
    """Efficient data loader for HuggingFace datasets."""
    
    __version__ = "1.0.0"  # Add version attribute
    
    def __init__(
        self,
        dataset_config: DatasetConfig,
        *,  # Force named parameters
        tokenizer: Optional[Union[PreTrainedTokenizer, PreTrainedTokenizerFast]] = None,
        device: Optional[str] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        use_auth_token: bool = False,
        max_length: Optional[int] = None
    ):
        """Initialize loader with configuration."""
        self.config = dataset_config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_auth_token = use_auth_token
        self.logger = get_logger(__name__)
        
        # Initialize and validate tokenizer
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(
            "facebook/bart-large",
            use_auth_token=use_auth_token
        )
        validate_tokenizer(self.tokenizer)
        
        # Set max_length based on tokenizer limits
        self.max_length = min(
            max_length or self.tokenizer.model_max_length,
            self.tokenizer.model_max_length
        )
        
        if self.config.cache_dir:
            self.config.cache_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_preset(
        cls,
        dataset_name: Literal["cnn_dailymail", "xsum"],
        **kwargs
    ) -> 'HuggingFaceLoader':
        """Create loader from preset configuration."""
        if dataset_name not in DATASET_CONFIGS:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        return cls(DATASET_CONFIGS[dataset_name], **kwargs)

    def load_and_preprocess(
        self,
        split: str = "train",
        *,  # Force named parameters
        max_length: Optional[int] = None,
        truncation: bool = True,
        padding: bool = True,
        use_cache: bool = True
    ) -> ProcessedDataset:
        """Load and preprocess dataset with caching."""
        max_length = max_length or self.max_length
        cache_path = self._get_cache_path(split, max_length, truncation, padding)
        
        # Try loading from cache
        if use_cache:
            cached_data = self._load_from_cache(cache_path)
            if cached_data:
                return cached_data

        # Load and process dataset
        try:
            dataset = load_dataset(
                self.config.dataset_name,
                split=split,
                cache_dir=self.config.cache_dir,
                use_auth_token=self.use_auth_token
            )
            
            processed_dataset = process_dataset_in_batches(
                dataset,
                tokenizer=self.tokenizer,
                text_column=self.config.text_column,
                summary_column=self.config.summary_column,
                batch_size=self.batch_size,
                max_length=max_length,
                truncation=truncation,
                padding=padding
            )
            
            # Cache the processed dataset
            if use_cache and cache_path:
                self._save_to_cache(processed_dataset, cache_path)
            
            return processed_dataset

        except ValueError as e:
            raise DatasetError(f"Invalid dataset configuration: {str(e)}") from e
        except FileNotFoundError as e:
            raise DatasetError(f"Dataset files not found: {str(e)}") from e
        except DatasetError as e:
            raise e
        except Exception as e:
            raise DatasetError(f"Dataset processing failed: {str(e)}") from e

    def _get_cache_path(
        self,
        split: str,
        max_length: Optional[int],
        truncation: bool,
        padding: bool
    ) -> Optional[Path]:
        """Generate cache path based on parameters."""
        if not self.config.cache_dir:
            return None
            
        # Create unique cache key
        params = f"{split}_{max_length}_{truncation}_{padding}"
        cache_key = hashlib.md5(params.encode()).hexdigest()
        
        return self.config.cache_dir / f"{self.config.dataset_name}_{cache_key}.cache"

    def _load_from_cache(
        self,
        cache_path: Optional[Path]
    ) -> Optional[ProcessedDataset]:
        """Load dataset from cache if available."""
        if not cache_path or not cache_path.exists():
            return None
            
        try:
            with cache_path.open('rb') as f:
                cached = torch.load(f)
                
            # Verify cache version and expiry
            if (
                cached.get('version') != self.__class__.__version__
                or (datetime.now() - cached['timestamp']).days > 7
            ):
                return None
                
            return ProcessedDataset(
                dataset=cached['dataset'],
                num_samples=cached['num_samples'],
                processing_time=cached['processing_time'],
                raw_texts=cached['raw_texts'],
                raw_summaries=cached['raw_summaries'],
                cache_hit=True,
                metadata=cached['metadata']
            )
        except FileNotFoundError as e:
            self.logger.warning(f"Cache file not found: {e}")
            return None
        except (KeyError, TypeError) as e:
            self.logger.warning(f"Cache format error: {e}")
            return None
        except torch.serialization.pickle.UnpicklingError as e:
            self.logger.warning(f"Cache unpickling error: {e}")
            return None
        except IOError as e:
            self.logger.warning(f"Cache I/O error: {e}")
            return None

    def _save_to_cache(
        self,
        dataset: ProcessedDataset,
        cache_path: Path
    ) -> None:
        """Save processed dataset to cache."""
        cache_data = {
            'version': self.__class__.__version__,
            'timestamp': datetime.now(),
            'dataset': dataset.dataset,
            'num_samples': dataset.num_samples,
            'processing_time': dataset.processing_time,
            'raw_texts': dataset.raw_texts,
            'raw_summaries': dataset.raw_summaries,
            'metadata': dataset.metadata
        }
        
        try:
            with cache_path.open('wb') as f:
                torch.save(cache_data, f)
        except IOError as e:
            self.logger.warning(f"Cache save failed - I/O error: {e}")
        except PermissionError as e:
            self.logger.warning(f"Cache save failed - permission denied: {e}")
        except Exception as e:
            self.logger.warning(f"Cache save failed: {e}")
