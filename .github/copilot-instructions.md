# Copilot Instructions for Text Summarization Project

## Description
This is a comprehensive Python project for text summarization and other NLP tasks, designed to leverage AWS infrastructure and parallel computing capabilities. Key features include:

- **Modular Design**: Well-structured tools for summarization, classification, and translation tasks
- **Parallel Processing**: Support for multi-GPU, single-GPU, and CPU environments
- **AWS Integration**: Optimized for AWS services (SageMaker, EC2, Lambda)
- **Model Support**: Integration with HuggingFace's transformers for:
    - Pre-trained model inference
    - Fine-tuning capabilities
    - Model parallel training
    - Distributed data processing

The codebase follows best practices for scalability, maintainability, and performance optimization, making it suitable for both research and production environments.

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
│ │ └── loader.py # Load datasets from HuggingFace or S3
│ │
│ ├── models/ # Summarization models
│ │ ├── extractive.py # Extractive summarization models
│ │ ├── abstractive.py # Abstractive summarization models
│ │ └── fine_tuning.py # Fine-tuning scripts
│ │
│ ├── evaluation/ # Model evaluation
│ │ └── metrics.py # ROUGE, BLEU, and other metrics
│ │
│ └── utils/ # Utility functions
│     ├── logging.py # Structured logging
│     └── aws_utils.py # AWS-specific utilities
│
├── configs/ # Configuration files
│ ├── model_configs/ # Model parameters
│ └── aws_configs/ # AWS configurations
│
├── tests/ # Unit and integration tests
├── scripts/ # Shell scripts for automation
├── requirements.txt # Python dependencies
└── README.md # Project documentation
```

## Code Style and Patterns

### 1. Code Organization 🏗️
- Write modular, single-responsibility components
- Prefer pure functions over classes
- Use dataclasses for structured and complex data
- NEVER use dictionaries, tuples, or named tuples - always use dataclasses instead
- Write the top-level function first, then implement helper functions

```python
from dataclasses import dataclass

@dataclass
class TokenSequence:
    text: str
    tokens: list[str]
    attention_score: float = 0.0
```

### General Patterns
```python
# ✅ DO: Use type hints and dataclasses for structured data
@dataclass
class SummarizationResult:
    summary: str
    rouge_score: float
    processing_time: float
    
# ❌ DON'T: Use dictionaries for results
result = {
    "summary": "The text summary...",
    "rouge_score": 0.85,
    "processing_time": 1.2
}
```

### Naming Conventions
- Models: `{type}_{purpose}_model` (e.g., `extractive_ranking_model`)
- Functions: verb_noun format (e.g., `process_text`, `generate_summary`)
- Parameters: descriptive nouns (e.g., `batch_size`, `learning_rate`)

### Function Patterns
```python
# ✅ DO: Use descriptive type hints and docstrings
def process_text(
    text: str,
    *,  # Force keyword arguments
    max_length: int,
    min_length: int,
    logger: Logger
) -> ProcessedTextResult:
    """
    Process input text for summarization.
    
    Args:
        text: Input text to process
        max_length: Maximum length of processed text 
        min_length: Minimum length of processed text
        remove_stopwords: Whether to remove stopwords
        language: Language code for processing
        logger: Logger for tracking

    Returns:
        ProcessedTextResult containing the cleaned text
    """
    
# ❌ DON'T Use config objects or dictionaries for parameters. Always use explicit arguments.
def process_text(text: str, config: ProcessingConfig) -> str:
    pass
```

### Model Handling
```python
# ✅ DO: Use explicit parameters
def initialize_model(
    model_name: str,
    *,
    device_map: str = "auto",
    use_bfloat16: bool = True,
    max_length: int = 512
) -> PreTrainedModel:
    model = AutoModel.from_pretrained(
        model_name,
        device_map=device_map,
        torch_dtype=torch.bfloat16 if use_bfloat16 else torch.float32,
        max_length=max_length
    )
    return model

# ❌ DON'T Use config objects. NEVER!
def initialize_model(config: ModelConfig) -> PreTrainedModel:
    pass
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

### Function Design Rules
- One function = one task
- Maximum 20 lines per function
- Maximum 2 levels of nesting
- Always use type hints
- NEVER TAKE THE INPUTS AS A CONFIG OBJECT

```python
def calculate_gene_score(
    gene_id: str,
    expression_data: np.ndarray,
    *,  # Force keyword arguments
    threshold: float = 0.05,
    logger: Logger
) -> CalculateGeneScoreResult:
```

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
) -> CalculateRougeResult:
    """Calculate ROUGE scores for predictions."""
```

### Logging Pattern
```python
# ✅ DO: Create a centralized logger in utils/logging.py
from typing import Any
import logging
import json

def get_logger(name: str = "summarization") -> logging.Logger:
    """Get or create a logger with consistent configuration."""
    logger = logging.getLogger(name)
    
    # Only add handler if logger doesn't have one
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            json.dumps({
                "message": "%(message)s",
                "level": "%(levelname)s"
            })
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    return logger

# Usage in other files:
from utils.logging import get_logger
logger = get_logger()
logger.info(json.dumps({
    "event": "batch_processing",
    "batch_size": len(batch),
    "memory_reserved": torch.cuda.memory_reserved()
}))
```

This approach ensures consistent logging configuration across all modules

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

### Documentation for Non-Python Developers
Documentation should be written assuming the reader has no Python experience:

1. **Project Overview**
   - Start with a high-level explanation of what the project does
   - Provide real-world examples of use cases
   - Include screenshots or diagrams when possible

2. **Setup Guide**
   - Step-by-step installation instructions
   - Required third-party tools and accounts
   - Common troubleshooting tips

3. **API Documentation**
   - Example requests and responses
   - Clear explanations of input/output formats
   - Error messages and their meanings

### Function Documentation
```python
def group_genes(
    expression_data: np.ndarray,
    *,
    min_size: int = 10,
    logger: Logger
) -> list[GeneGroup]:
    """Group genes based on expression patterns.

    Args:
        expression_data: Gene expression matrix (genes × samples)
        min_size: Minimum genes per group (default: 10)
        logger: Logger instance for tracking

    Returns:
        List of GeneGroup objects
    """
```

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


### Class Documentation
```python
class SummarizationPipeline:
    """
    End-to-end pipeline for text summarization.
    
    Attributes:
        model: The underlying summarization model
        tokenizer: Tokenizer for text processing
        
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
