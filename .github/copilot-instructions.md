# Copilot Instructions for Text Summarization Project

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
# ✅ DO: Use structured logging
logger.info("Processing batch", extra={
    "batch_size": len(batch),
    "memory_used": torch.cuda.memory_allocated()
})

# ❌ DON'T: Use print statements or f-strings for logging
print(f"Processing batch of size {len(batch)}")
```

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