"""Module for all dataclasses used in the summarization pipeline."""

from dataclasses import dataclass
from typing import Any, List, Literal, Optional


@dataclass
class LoadTokenizerResult:
    tokenizer: Any
    special_tokens_dict: dict
    special_tokens_added: bool
    
@dataclass
class PreprocessedDataset:
    train_dataset: Any
    test_dataset: Any
    val_dataset: Any

@dataclass
class TrainModelResult:
    model: Any
    training_args: Any

@dataclass
class Metrics:
    # Basic evaluation metrics
    f1: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    sensitivity: Optional[float] = None
    specificity: Optional[float] = None
    accuracy: Optional[float] = None
    # Summarization-specific metrics
    rouge: Optional[float] = None
    bleu: Optional[float] = None
    bert_score: Optional[float] = None
    meteor: Optional[float] = None
    rouge_1: Optional[float] = None
    rouge_2: Optional[float] = None
    rouge_L: Optional[float] = None

@dataclass
class EvaluationResult:
    metrics: Metrics

@dataclass
class PipelineResult:
    train_model_result: TrainModelResult
    evaluation_metrics: Any

@dataclass
class DatasetConfig:
    dataset_name: Literal["cnn_dailymail", "xsum"]
    dataset_version: str
    sample_size: int
    input_column: str
    target_column: str

@dataclass
class ModelConfig:
    model_type: str
    tokenizer_name: Literal["facebook/bart-large-cnn", 
                            "bert-base-uncased",
                            "distilbert-base-uncased",
                            "facebook/bart-base",
                            "t5-small"]
    pretrained_model_name: str

@dataclass
class TrainingConfig:
    batch_size: int
    learning_rate: float
    num_epochs: int

@dataclass
class EvaluationConfig:
    metrics: List[str]

@dataclass
class OutputConfig:
    output_dir: str

@dataclass
class Config:
    dataset: DatasetConfig
    model: ModelConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    output: OutputConfig

@dataclass
class LoadDatasetResult:
    dataset: Any
    train_dataset: Any
    test_dataset: Any
    val_dataset: Any
