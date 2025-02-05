"""Custom exceptions for the summarization pipeline."""

class SummarizationError(Exception):
    """Base class for summarization-specific errors."""
    pass

class TextTooLongError(SummarizationError):
    """Error raised when text is too long to process."""
    pass

class TokenLengthError(SummarizationError):
    """Raised when input text exceeds model's maximum token length."""
    pass

class BatchProcessingError(SummarizationError):
    """Raised when batch processing fails."""
    pass
