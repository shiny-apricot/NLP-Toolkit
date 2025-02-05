"""Summary evaluation metrics."""

from dataclasses import dataclass
from typing import List, Dict
from rouge_score import rouge_scorer
import numpy as np

@dataclass
class EvaluationMetrics:
    """Evaluation metrics for summarization."""
    rouge_scores: Dict[str, float]
    compression_ratios: List[float]
    processing_times: List[float]
    
    @property
    def mean_compression_ratio(self) -> float:
        return np.mean(self.compression_ratios)
    
    @property
    def mean_processing_time(self) -> float:
        return np.mean(self.processing_times)

def calculate_metrics(
    predictions: List[str],
    references: List[str],
    processing_times: List[float]
) -> EvaluationMetrics:
    """Calculate evaluation metrics."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    scores = [scorer.score(ref, pred) for ref, pred in zip(references, predictions)]
    
    rouge_scores = {
        'rouge1': np.mean([s['rouge1'].fmeasure for s in scores]),
        'rouge2': np.mean([s['rouge2'].fmeasure for s in scores]),
        'rougeL': np.mean([s['rougeL'].fmeasure for s in scores])
    }
    
    compression_ratios = [
        len(pred.split()) / len(ref.split())
        for pred, ref in zip(predictions, references)
    ]
    
    return EvaluationMetrics(
        rouge_scores=rouge_scores,
        compression_ratios=compression_ratios,
        processing_times=processing_times
    )
