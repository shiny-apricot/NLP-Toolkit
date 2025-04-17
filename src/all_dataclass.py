"""Module for all dataclasses used in the summarization pipeline."""

from dataclasses import dataclass
from typing import Any, List, Literal, Optional, Dict


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
    max_input_length: int                          # Maximum length of input text
    max_target_length: int                         # Maximum length of target summaries
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
    gradient_accumulation_steps: int = 1  # Number of steps to accumulate gradients
    mixed_precision: bool = False  # Whether to use mixed precision training
    eval_steps: int = 500  # Evaluate model every n steps

@dataclass
class HardwareConfig:
    """Configuration for hardware-specific settings.
    
    Defines parameters specific to the hardware being used for training,
    such as GPU memory utilization and random seed.
    """
    gpu_memory_utilization: float = 0.9  # Percentage of GPU memory to use
    seed: int = 42  # Random seed for reproducibility

@dataclass
class EvaluationConfig:
    """Configuration for model evaluation.
    
    Specifies which metrics to calculate during evaluation.
    """
    metrics: List[str]  # List of metric names to calculate
    eval_batch_size: int = 16  # Batch size for evaluation

@dataclass
class OutputConfig:
    """Configuration for output management.
    
    Defines the directory where output files (e.g., model checkpoints,
    evaluation results) will be saved and whether to save the model.
    """
    output_dir: str   # Directory to save output files
    save_model: bool = True  # Whether to save the trained model
    save_steps: int = 1000  # Save model checkpoint every n steps
    logging_steps: int = 100  # Log training metrics every n steps

@dataclass
class Config:
    """Complete configuration for the summarization pipeline.
    
    Aggregates all individual configurations for dataset, model, training,
    evaluation, output management into a single configuration object.
    """
    dataset: DatasetConfig    # Configuration for dataset selection and processing
    model: ModelConfig        # Configuration for model selection and initialization
    training: TrainingConfig  # Configuration for model training
    evaluation: EvaluationConfig  # Configuration for model evaluation
    output: OutputConfig      # Configuration for output management
    hardware: Optional[HardwareConfig] = None  # Hardware-specific configuration

@dataclass
class DatasetStatistics:
    """Statistics about dataset texts and their lengths."""
    article_max_length: int
    article_min_length: int
    article_avg_length: float
    article_median_length: float
    article_std_length: float
    article_max_char_length: int
    article_avg_char_length: float
    
    summary_max_length: int
    summary_min_length: int
    summary_avg_length: float
    summary_median_length: float
    summary_std_length: float
    summary_max_char_length: int
    summary_avg_char_length: float
    
    article_length_distribution: Dict[str, int]
    summary_length_distribution: Dict[str, int]
    
    dataset_size: int
    compression_ratio: float  # avg(summary_length/article_length)

@dataclass
class LoadDatasetResult:
    """Result of loading and preprocessing a dataset."""
    dataset: Any
    train_dataset: Any
    test_dataset: Any
    val_dataset: Any
    train_statistics: DatasetStatistics = None  # type: ignore
    test_statistics: DatasetStatistics = None  # type: ignore
    validation_statistics: DatasetStatistics = None  # type: ignore
    full_train_statistics: DatasetStatistics = None  # type: ignore
    full_test_statistics: DatasetStatistics = None  # type: ignore
    full_validation_statistics: DatasetStatistics = None  # type: ignore

@dataclass
class LoadPretrainedModelResult:
    """Results from loading a pre-trained model.
    
    Contains the loaded model and configuration for evaluation.
    """
    model: Any           # The loaded pre-trained model
    model_config: Any    # Configuration of the loaded model

@dataclass
class PretrainedEvalResult:
    """Complete results from running the pre-trained model evaluation pipeline.
    
    Aggregates the results from loading the pre-trained model, evaluation,
    and saving steps to provide a comprehensive view of the pipeline execution outcome.
    """
    model_result: LoadPretrainedModelResult  # Results from loading the pre-trained model
    evaluation_metrics: Metrics              # Metrics from the evaluation stage
    save_results: SaveResultsOutput          # Results from saving the evaluation output
