"""Module for all dataclasses used in the summarization pipeline."""

from dataclasses import dataclass
from typing import Any, List

from main import TrainModelResult

@dataclass
class PipelineResult:
    train_model_result: TrainModelResult
    evaluation_metrics: Any

@dataclass
class DatasetConfig:
    dataset_name: str
    sample_size: int

@dataclass
class ModelConfig:
    model_type: str
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