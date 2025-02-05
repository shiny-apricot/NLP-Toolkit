"""CNN-Daily Mail dataset handler module."""

from dataclasses import dataclass
from typing import List, Tuple, Optional
from datasets import load_dataset, Dataset
import numpy as np

@dataclass
class CNNDailyArticle:
    """Single article from CNN-Daily dataset."""
    article: str
    highlights: str
    article_id: str
    article_length: int
    highlight_length: int

@dataclass
class DatasetSplit:
    """Represents a subset of the dataset."""
    articles: List[CNNDailyArticle]
    size: int
    split_name: str

def load_cnn_daily_dataset(
    *,
    split: str = "train",
    sample_size: Optional[int] = None,
    random_seed: int = 42
) -> DatasetSplit:
    """Load CNN-Daily Mail dataset.
    
    Args:
        split: Dataset split to load ("train", "validation", "test")
        sample_size: If provided, randomly sample this many articles
        random_seed: Random seed for sampling
    """
    dataset = load_dataset("cnn_dailymail", "3.0.0", split=split)
    
    if sample_size:
        dataset = dataset.shuffle(seed=random_seed)
        dataset = dataset.select(range(min(sample_size, len(dataset))))

    articles = [
        CNNDailyArticle(
            article=item["article"],
            highlights=item["highlights"],
            article_id=str(idx),
            article_length=len(item["article"].split()),
            highlight_length=len(item["highlights"].split())
        )
        for idx, item in enumerate(dataset)
    ]

    return DatasetSplit(
        articles=articles,
        size=len(articles),
        split_name=split
    )
