# Copilot Instructions for Text Summarization Project

## Description

This is a project to help us on our research projects. 
The goal is to have some useful tools to use while working on parallel computation (multiple GPUs) and LLM models.
Help me to write good and modular tools that I can use in different projects (especially the current summarization task.)
Know that I will use AWS resources to run the code, so make sure to write code that can be run on AWS.
This is a project not only for summarization but also for other NLP tasks. (e.g., text classification, translation, etc.)
I will heavily utilize HuggingFace models and datasets.
Write code that can be run on multiple GPUs, single GPU and CPU.

## Usage Scenarios

This project will be used in the following contexts:
- **Batch Processing on AWS**: Running large-scale summarization on datasets.
- **Interactive Research Notebooks**: Running in Jupyter or Colab for experimentation.

Ensure the code can be easily adapted to these different use cases.

## AWS-Specific Considerations

We will be using AWS for deployment. Keep the following in mind:
- **S3 for Storage**: Use S3 for dataset storage and model checkpoints.
- **SageMaker for Training**: Model training will use AWS SageMaker.
- **Inference on Lambda or EC2**: Choose between cost-effective Lambda for small tasks or EC2 for batch processing.
- **Cost Optimization**: Prefer `bfloat16` over `float32` to reduce memory usage, and always clean up unused resources.

## Project Structure
```
summarization_project/
├── src/ # Core project source code
│ ├── data/ # Data handling and preprocessing
│ │ ├── preprocessing.py # Text cleaning, tokenization, and normalization
│ │ ├── loader.py # Load datasets from HuggingFace or S3
│ │ └── augmentation.py # Data augmentation techniques (e.g., back translation)
│ │
│ ├── models/ # Summarization models
│ │ ├── extractive_sum/ # Extractive summarization models (e.g., BERT, TextRank)
│ │ │ ├── bert_extractive_model.py
│ │ │ └── textrank_model.py
│ │ ├── abstractive_sum/ # Abstractive summarization models (e.g., T5, GPT, LLaMA)
│ │ │ ├── t5_abstractive_model.py
│ │ │ ├── bart_abstractive_model.py
│ │ │ └── gpt_abstractive_model.py
│ │ ├── fine_tuning/ # Fine-tuning scripts for models
│ │ │ ├── fine_tune_bart.py
│ │ │ ├── fine_tune_t5.py
│ │ │ └── fine_tune_gpt.py
│ │ └── pipelines.py # Unified pipelines for extractive and abstractive summarization
│ │
│ ├── evaluation/ # Model evaluation
│ │ ├── metrics.py # ROUGE, BLEU, and METEOR metrics
│ │ ├── analysis.py # Error analysis and visualization
│ │ └── reporting.py # Generate evaluation reports
│ │
│ ├── utils/ # Utility functions
│ │ ├── logging.py # Structured logging with AWS integration
│ │ ├── aws_utils.py # AWS-specific utilities (S3, SageMaker, etc.)
│ │ └── gpu_utils.py # GPU resource management
│ │
│ └── cache/ # Cache for trained models
│ │ ├── bart/ # Cached BART models
│ │ ├── t5/ # Cached T5 models
│ │ └── gpt/ # Cached GPT models
│
├── configs/ # Configuration files
│ ├── model_configs/ # Model-specific configurations
│ ├── aws_configs/ # AWS deployment configurations
│ └── prompt_configs/ # Prompt engineering configurations
│
├── tests/ # Unit and integration tests
│ ├── test_data.py # Test data loading and preprocessing
│ ├── test_models.py # Test summarization models
│ └── test_evaluation.py # Test evaluation metrics
│
├── scripts/ # Shell scripts for automation
│ ├── train.sh # Train summarization models
│ ├── evaluate.sh # Evaluate models
│ └── deploy_aws.sh # Deploy models on AWS
│
├── requirements.txt # Python dependencies
├── Dockerfile # Docker configuration for containerization
├── .gitignore # Files to ignore in version control
├── .env # Environment variables
└── README.md # Project documentation
```


## Code Style and Patterns

### General Patterns
```python
# ✅ DO: Use type hints and dataclasses for structured data
@dataclass
class SummarizationConfig:
    max_length: int
    min_length: int
    model_name: str
    
# ❌ DON'T: Use dictionaries for configuration
config = {
    "max_length": 512,
    "min_length": 50
}
```

### Naming Conventions
- Models: `{type}_{purpose}_model` (e.g., `extractive_ranking_model`)
- Configs: `{component}Config` (e.g., `TokenizerConfig`)
- Functions: verb_noun format (e.g., `process_text`, `generate_summary`)
- Parameters: descriptive nouns (e.g., `batch_size`, `learning_rate`)

### Function Patterns
```python
# ✅ DO: Use descriptive type hints and docstrings
def process_article(
    text: str,
    config: ProcessingConfig,
    logger: Logger
) -> ProcessedText:
    """
    Process article text for summarization.
    
    Args:
        text: Raw article text
        config: Processing configuration
        logger: Logger instance
    
    Returns:
        ProcessedText object containing cleaned text
    """
    
# ❌ DON'T: Use generic parameter names or skip documentation
def process(text, cfg):
    pass
```

## Project-Specific Patterns

### Data Loading
```python
# ✅ DO: Use context managers and proper error handling
def load_dataset(path: Path) -> Dataset:
    try:
        with open(path, 'r') as f:
            # processing
    except FileNotFoundError:
        logger.error(f"Dataset not found: {path}")
        raise
```

### Model Handling
```python
# ✅ DO: Use proper model initialization pattern
def initialize_model(config: ModelConfig) -> PreTrainedModel:
    model = AutoModel.from_pretrained(
        config.model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    return model

# ❌ DON'T: Hardcode model parameters
model = AutoModel.from_pretrained("t5-base")
```

### Prompt Engineering
```python
# ✅ DO: Use template strings for prompts
SUMMARY_TEMPLATE = """
Summarize the following article:
{text}
Key points to include:
{key_points}
"""

# ❌ DON'T: Use string concatenation
prompt = "Summarize: " + text + " Key points: " + key_points
```

## Common Components

### Data Processing
```python
@dataclass
class TextProcessor:
    tokenizer: PreTrainedTokenizer
    max_length: int
    
    def __post_init__(self):
        self.validate_config()
    
    def validate_config(self):
        if self.max_length > self.tokenizer.model_max_length:
            raise ValueError("max_length exceeds model capacity")
```

### Model Evaluation
```python
def calculate_rouge(
    predictions: List[str],
    references: List[str],
    rouge_types: List[str] = ["rouge1", "rouge2", "rougeL"]
) -> Dict[str, float]:
    """Calculate ROUGE scores for predictions."""
```

### Logging Pattern
```python
# ✅ DO: Use structured logging with AWS integration
import logging
import json

logger = logging.getLogger("summarization")
handler = logging.StreamHandler()
formatter = logging.Formatter(json.dumps({"message": "%(message)s", "level": "%(levelname)s"}))
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

logger.info(json.dumps({
    "event": "batch_processing",
    "batch_size": len(batch),
    "memory_reserved": torch.cuda.memory_reserved()
}))
```

- Avoid multiple logging initialization in different modules to prevent duplicate logs.

## AWS-Specific Patterns

### Resource Management
```python
# ✅ DO: Use resource cleanup patterns
class GPUManager:
    def __enter__(self):
        torch.cuda.empty_cache()
        return self
    
    def __exit__(self, *args):
        torch.cuda.empty_cache()
```

### Checkpointing
```python
def save_checkpoint(
    model: PreTrainedModel,
    optimizer: Optimizer,
    epoch: int,
    path: Path
):
    """Save training checkpoint with proper error handling."""
```

## Testing Patterns

```python
# ✅ DO: Write testable functions with clear inputs/outputs
def extract_key_sentences(
    text: str,
    num_sentences: int = 3
) -> List[str]:
    """Extract key sentences for testing."""
    return sentences

# ❌ DON'T: Mix processing and side effects
def process_and_save(text):
    sentences = extract_sentences(text)
    save_to_file(sentences)  # Side effect makes testing difficult
```

## Documentation Requirements

### File Documentation
At the top of each Python file, include a brief description of the file's purpose and contents.
```python
"""
data_loader.py
Module for loading and preprocessing text data for summarization.
This module provides functions to load text data from various sources and preprocess it for model input.
Functions:
    - load_dataset: Load text data from a file or database
    - preprocess_text: Clean and tokenize text data
"""
```
This is just an example... I will write the actual description when I see the code.

### Function Documentation
```python
def fine_tune_model(
    model: PreTrainedModel,
    dataset: Dataset,
    config: TrainingConfig
) -> Tuple[PreTrainedModel, Dict[str, float]]:
    """
    Fine-tune a pre-trained model on the summarization dataset.
    
    Args:
        model: The pre-trained model to fine-tune
        dataset: Training dataset
        config: Training configuration
        
    Returns:
        Tuple of (fine-tuned model, training metrics)
        
    Raises:
        OutOfMemoryError: If batch size is too large
        ValueError: If dataset is empty
    """
```

### Class Documentation
```python
class SummarizationPipeline:
    """
    End-to-end pipeline for text summarization.
    
    Attributes:
        model: The underlying summarization model
        tokenizer: Tokenizer for text processing
        config: Pipeline configuration
        
    Example:
        >>> pipeline = SummarizationPipeline(model_name="t5-base")
        >>> summary = pipeline.summarize("Long article text...")
    """
```

## Error Handling

```python
# ✅ DO: Use specific error types and proper handling
class SummarizationError(Exception):
    """Base class for summarization-specific errors."""
    pass

class TokenLengthError(SummarizationError):
    """Raised when text exceeds model's maximum token length."""
    pass

# ❌ DON'T: Use bare except clauses
try:
    process_text()
except Exception as e:  # Too broad
    pass
```

## Security Considerations

- **IAM Role Management**: Ensure models have the least privileges necessary.
- **Avoid Hardcoding Secrets**: Use AWS Secrets Manager for API keys.
- **Data Privacy**: If using user-generated content, consider encryption before storing results.
