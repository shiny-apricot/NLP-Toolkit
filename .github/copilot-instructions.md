# Text Summarization MVP - Coding Standards

## Core Principles

- Use pure functions over classes
- Use dataclasses for all data structures
- Always use type hints
- Keep functions under 20 lines
- Functions must take parameters one by one (no grouped arguments)
- Functions must always return a single dataclass
- Avoid unnecessary complexity; keep the project simple and understandable
- While writing docs and variable names, consider the reader as a new team member

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
    text: str
) -> SummarizationResult:
    """Process input text for summarization.
    
    Args:
        text: Input text to process
    
    Returns:
        SummarizationResult dataclass with processed text
    """
    # ...function logic...
```

### Data Processing
```python
@dataclass
class CNNDailyMailExample:
    article: str
    highlights: str
    article_tokens: list[str]

def load_cnn_dataset(
    sample_size: int
) -> list[CNNDailyMailExample]:
    """Load and sample CNN/Daily Mail dataset.
    
    Args:
        sample_size: Number of samples to load
    
    Returns:
        List of CNNDailyMailExample dataclasses
    """
    # ...function logic...
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
