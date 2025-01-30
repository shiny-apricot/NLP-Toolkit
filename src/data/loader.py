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

@dataclass
class DatasetConfig:
    """Configuration for dataset loading."""
    dataset_name: Optional[str] = None
    s3_path: Optional[str] = None
    cache_dir: Optional[str] = None
    split: str = "train"
    streaming: bool = False
    text_column: str = "text"
    summary_column: str = "summary"
    max_samples: Optional[int] = None

class DataLoader:
    """Handle dataset loading from various sources."""
    
    def __init__(
        self,
        config: DatasetConfig,
        preprocessing_config: Optional[PreprocessingConfig] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        logger: Optional[logging.Logger] = None
    ):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.preprocessor = TextPreprocessor(
            preprocessing_config or PreprocessingConfig(
                max_length=512,
                min_length=50,
                model_name="t5-base"
            ),
            tokenizer=tokenizer
        )
        self.s3_client = boto3.client('s3') if config.s3_path else None

    def load_from_huggingface(self) -> Union[Dataset, DatasetDict]:
        """Load dataset from HuggingFace Hub."""
        if not self.config.dataset_name:
            raise ValueError("Dataset name must be provided for HuggingFace loading")

        try:
            dataset = load_dataset(
                self.config.dataset_name,
                split=self.config.split,
                streaming=self.config.streaming,
                cache_dir=self.config.cache_dir
            )
            
            if self.config.max_samples:
                dataset = dataset.select(range(self.config.max_samples))
                
            return dataset
        except Exception as e:
            self.logger.error(f"Failed to load dataset from HuggingFace: {str(e)}")
            raise

    def load_from_s3(self) -> Dataset:
        """Load dataset from AWS S3."""
        if not self.config.s3_path:
            raise ValueError("S3 path must be provided for S3 loading")

        try:
            bucket, key = self._parse_s3_path(self.config.s3_path)
            response = self.s3_client.get_object(Bucket=bucket, Key=key)
            data = json.load(response['Body'])
            
            # Convert to HuggingFace dataset format
            dataset = Dataset.from_dict(data)
            
            if self.config.max_samples:
                dataset = dataset.select(range(self.config.max_samples))
                
            return dataset
        except Exception as e:
            self.logger.error(f"Failed to load dataset from S3: {str(e)}")
            raise

    def _parse_s3_path(self, s3_path: str) -> tuple:
        """Parse S3 path into bucket and key."""
        path = s3_path.replace("s3://", "")
        bucket, *key_parts = path.split("/")
        key = "/".join(key_parts)
        return bucket, key

    def preprocess_dataset(
        self,
        dataset: Dataset,
        batch_size: int = 32
    ) -> Dataset:
        """Preprocess loaded dataset."""
        
        def process_batch(examples):
            texts = examples[self.config.text_column]
            summaries = examples[self.config.summary_column]
            
            # Preprocess inputs
            input_tokens = self.preprocessor.batch_process(texts, batch_size)
            
            # Preprocess targets/summaries if available
            target_tokens = self.preprocessor.batch_process(summaries, batch_size)
            
            return {
                "input_ids": input_tokens[0]["input_ids"],
                "attention_mask": input_tokens[0]["attention_mask"],
                "labels": target_tokens[0]["input_ids"]
            }

        return dataset.map(
            process_batch,
            batched=True,
            batch_size=batch_size,
            remove_columns=dataset.column_names
        )

    def __call__(self) -> Dataset:
        """Load and preprocess dataset."""
        # Load from appropriate source
        if self.config.dataset_name:
            dataset = self.load_from_huggingface()
        elif self.config.s3_path:
            dataset = self.load_from_s3()
        else:
            raise ValueError("Either dataset_name or s3_path must be provided")

        # Preprocess dataset
        return self.preprocess_dataset(dataset)

    def get_data_collator(self):
        """Return a data collator function for batching."""
        def collate_fn(examples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
            # Combine all tensors in the batch
            return {
                key: torch.stack([example[key] for example in examples])
                for key in examples[0].keys()
            }
        
        return collate_fn
