# Text Summarization MVP - Coding Standards

## Core Principles

- Write pure functions over classes
- Use dataclasses for data structures
- Always use type hints
- Keep functions under 20 lines
- Never use Config classes for input parameters
- While writing docs and variable names, consider the reader as a new team member

## Project Organization

### Essential Directory Structure
```
summarization-project/
├── src/
│   ├── data_processing/       # Dataset handling
│   ├── models/                # BART and TextRank implementations
│   ├── evaluation/            # Evaluation metrics
│   └── utils/                 # Shared utilities
├── tests/                     # Test files
└── configs/                   # Configuration files
```

## Key Coding Patterns

### Data Structures
```python
@dataclass
class SummarizationResult:
    summary: str
    rouge_score: float
    processing_time: float
```

### Function Pattern
```python
def process_text(
    text: str,
    *,  # Force keyword arguments
    max_length: int,
    min_length: int,
    logger: Logger
) -> ProcessedTextResult:
    """Process input text for summarization.
    
    Args:
        text: Input text to process
        max_length: Maximum length of processed text
        min_length: Minimum length of processed text
        logger: Logger instance
    
    Returns:
        ProcessedTextResult with processed text
    """
```

### Data Processing
```python
@dataclass
class CNNDailyMailExample:
    article: str
    highlights: str
    article_tokens: list[str]
    
def load_cnn_dataset(
    *,
    sample_size: int,
    random_seed: int = 42
) -> list[CNNDailyMailExample]:
    """Load and sample CNN/Daily Mail dataset."""
```

### Error Handling
```python
class SummarizationError(Exception):
    """Base class for summarization errors."""
    pass

class TokenLengthError(SummarizationError):
    """Error when text exceeds token limits."""
    pass

# ✅ DO: Handle specific exceptions
try:
    process_text(text, max_length=512, min_length=50, logger=logger)
except TokenLengthError as e:
    logger.error(f"Text too long: {e}")

# ❌ DON'T: Use bare except
try:
    process_text()
except Exception:  # Too broad
    pass
```

### Evaluation
```python
@dataclass
class EvaluationMetrics:
    rouge1: float
    rouge2: float
    rougeL: float
    processing_time: float

def calculate_metrics(
    predictions: list[str],
    references: list[str],
    *,
    metrics: list[str] = ["rouge1", "rouge2", "rougeL"]
) -> EvaluationMetrics:
    """Calculate evaluation metrics for predictions."""
```

## Testing Requirements

- Write unit tests for core functionality
- Use pytest for testing framework
- Mock external services like AWS

## AWS Guidelines

1. Resource Configuration
   - BART: Use SageMaker ml.p3.2xlarge instances
   - TextRank: Use CPU instances
   - Use S3 for storing datasets and results

2. Security Best Practices
   - No hardcoded credentials
   - Use IAM roles with least privilege

## Logging

```python
def setup_logger(
    name: str, 
    *,
    level: str = "INFO"
) -> Logger:
    """Set up a logger with consistent formatting."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger
```

## Documentation Requirements

Every function needs:
- Type hints
- Docstring with Args/Returns

Every file needs:
- Module docstring 
- Purpose description

## Version Control

- Use meaningful commit messages
- Keep PRs focused on single features/fixes
