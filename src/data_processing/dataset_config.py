"""Dataset configuration and validation."""

from dataclasses import dataclass
from typing import Optional
from pathlib import Path


@dataclass
class DatasetConfig:
    """Dataset configuration with validation."""
    dataset_name: str
    text_column: str
    summary_column: str
    cache_dir: Optional[Path] = None
    revision: str = "main"
    
    def __post_init__(self):
        if not self.dataset_name:
            raise ValueError("Dataset name cannot be empty")
        if not self.text_column:
            raise ValueError("Text column name cannot be empty")
        if not self.summary_column:
            raise ValueError("Summary column name cannot be empty")


# Predefined dataset configurations
DATASET_CONFIGS = {
    "cnn_dailymail": DatasetConfig(
        dataset_name="cnn_dailymail",
        text_column="article",
        summary_column="highlights"
    ),
    "xsum": DatasetConfig(
        dataset_name="xsum",
        text_column="document",
        summary_column="summary"
    )
}
