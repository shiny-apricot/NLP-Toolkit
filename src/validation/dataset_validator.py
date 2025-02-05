"""Dataset validation utilities."""

from dataclasses import dataclass
from typing import List
import statistics
from src.data.cnn_daily_dataset import DatasetSplit, CNNDailyArticle

@dataclass
class ValidationMetrics:
    """Dataset validation metrics."""
    total_articles: int
    avg_article_length: float
    avg_summary_length: float
    empty_articles: int
    empty_summaries: int
    length_distribution: dict

def validate_dataset_split(dataset: DatasetSplit) -> ValidationMetrics:
    """Validate dataset split and compute metrics."""
    article_lengths = [a.article_length for a in dataset.articles]
    summary_lengths = [a.highlight_length for a in dataset.articles]
    
    return ValidationMetrics(
        total_articles=dataset.size,
        avg_article_length=statistics.mean(article_lengths),
        avg_summary_length=statistics.mean(summary_lengths),
        empty_articles=sum(1 for a in dataset.articles if not a.article.strip()),
        empty_summaries=sum(1 for a in dataset.articles if not a.highlights.strip()),
        length_distribution={
            "article_length_percentiles": {
                f"p{p}": statistics.quantiles(article_lengths, n=100)[p-1]
                for p in [25, 50, 75, 90]
            },
            "summary_length_percentiles": {
                f"p{p}": statistics.quantiles(summary_lengths, n=100)[p-1]
                for p in [25, 50, 75, 90]
            }
        }
    )
