"""Module for all dataclasses used in the summarization pipeline."""

from dataclasses import dataclass
from typing import Any, List, Literal, Optional


@dataclass
class LoadTokenizerResult:
    """Results from loading and configuring a tokenizer.
    
    Contains the tokenizer object and information about any special tokens added.
    """
    tokenizer: Any  # The loaded tokenizer object
    special_tokens_dict: dict  # Dictionary of special tokens added to the tokenizer
    special_tokens_added: bool  # Flag indicating whether special tokens were added successfully
    
@dataclass
class PreprocessedDataset:
    """Container for preprocessed train, test, and validation datasets.
    
    Holds the split datasets after preprocessing steps have been applied.
    """
    train_dataset: Any  # The preprocessed training dataset
    test_dataset: Any   # The preprocessed test dataset
    val_dataset: Any    # The preprocessed validation dataset

@dataclass
class TrainModelResult:
    """Results from training a model.
    
    Contains the trained model and the training arguments used during training.
    This object is returned by training functions to provide access to the
    model for evaluation and inference.
    """
    model: Any           # The trained model object
    training_args: Any   # Configuration/arguments used during training

@dataclass
class Metrics:
    """Collection of evaluation metrics for model performance assessment.
    
    Contains both general machine learning metrics and 
    summarization-specific evaluation metrics. All fields are optional
    as different evaluation scenarios may calculate different metrics.
    """
    # Basic evaluation metrics
    f1: Optional[float] = None          # Harmonic mean of precision and recall
    precision: Optional[float] = None    # Ratio of true positives to all predicted positives
    recall: Optional[float] = None       # Ratio of true positives to all actual positives
    sensitivity: Optional[float] = None  # True positive rate (same as recall)
    specificity: Optional[float] = None  # True negative rate
    accuracy: Optional[float] = None     # Ratio of correct predictions to total predictions
    
    # Summarization-specific metrics
    rouge: Optional[float] = None        # Overall ROUGE score (if a single value is used)
    bleu: Optional[float] = None         # BLEU score for n-gram precision
    bert_score: Optional[float] = None   # Contextual embedding similarity using BERT
    meteor: Optional[float] = None       # Metric for evaluation of translation with explicit ordering
    rouge_1: Optional[float] = None      # ROUGE score for unigrams
    rouge_2: Optional[float] = None      # ROUGE score for bigrams
    rouge_L: Optional[float] = None      # ROUGE score based on longest common subsequence

@dataclass
class EvaluationResult:
    """Results from evaluating a trained model.
    
    Encapsulates the metrics calculated during model evaluation.
    Serves as a container for all performance metrics of interest.
    """
    metrics: Metrics     # Collection of calculated performance metrics

@dataclass
class SaveResultsOutput:
    """Results from saving evaluation metrics.
    
    Tracks the location where results were saved and whether the
    saving operation was successful.
    """
    file_path: str       # Path where the results were saved
    success: bool        # Whether the save operation completed successfully
    
@dataclass
class PipelineResult:
    """Complete results from running the full summarization pipeline.
    
    Aggregates the results from training, evaluation, and saving steps
    to provide a comprehensive view of the pipeline execution outcome.
    """
    train_model_result: TrainModelResult  # Results from the model training stage
    evaluation_metrics: Metrics           # Metrics from the evaluation stage
    save_results: SaveResultsOutput       # Results from saving the evaluation output

@dataclass
class DatasetConfig:
    """Configuration for dataset selection and processing.
    
    Defines which dataset to use, version information, sample size,
    and which columns contain the input text and target summaries.
    """
    dataset_name: Literal["cnn_dailymail", "xsum"]  # Name of the dataset to use
    dataset_version: str                            # Version/variant of the dataset
    sample_size: int                                # Number of examples to use
    input_column: str                               # Column name containing input text
    target_column: str                              # Column name containing target summaries

@dataclass
class ModelConfig:
    """Configuration for model selection and initialization.
    
    Specifies the model name to use for both the model and tokenizer,
    and the name of the pretrained model to load.
    """
    model_name: str  # Name of the model to use for both model and tokenizer
    pretrained_model_name: str  # Name of the pretrained model to load

@dataclass
class TrainingConfig:
    """Configuration for model training.
    
    Defines the training parameters such as batch size, learning rate,
    and the number of epochs to train for.
    """
    batch_size: int      # Number of samples per batch
    learning_rate: float # Learning rate for the optimizer
    num_epochs: int      # Number of epochs to train the model

@dataclass
class EvaluationConfig:
    """Configuration for model evaluation.
    
    Specifies which metrics to calculate during evaluation.
    """
    metrics: List[str]  # List of metric names to calculate

@dataclass
class OutputConfig:
    """Configuration for output management.
    
    Defines the directory where output files (e.g., model checkpoints,
    evaluation results) will be saved.
    """
    output_dir: str  # Directory to save output files

@dataclass
class Config:
    """Complete configuration for the summarization pipeline.
    
    Aggregates all individual configurations for dataset, model, training,
    evaluation, and output management into a single configuration object.
    """
    dataset: DatasetConfig    # Configuration for dataset selection and processing
    model: ModelConfig        # Configuration for model selection and initialization
    training: TrainingConfig  # Configuration for model training
    evaluation: EvaluationConfig  # Configuration for model evaluation
    output: OutputConfig      # Configuration for output management

@dataclass
class LoadDatasetResult:
    """Results from loading and splitting a dataset.
    
    Contains the full dataset and the split train, test, and validation datasets.
    """
    dataset: Any        # The full loaded dataset
    train_dataset: Any  # The training split of the dataset
    test_dataset: Any   # The test split of the dataset
    val_dataset: Any    # The validation split of the dataset
