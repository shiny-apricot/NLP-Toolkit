from dataclasses import dataclass
from typing import List, Dict, Optional
import torch
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
import numpy as np
import logging
from concurrent.futures import ThreadPoolExecutor
import nltk
from tqdm import tqdm

@dataclass
class MetricsResult:
    rouge_scores: Dict[str, float]
    bleu_score: float
    meteor_score: float
    mean_scores: Dict[str, float]

def calculate_rouge_scores(
    predictions: List[str],
    references: List[str],
    *,
    rouge_types: List[str] = None,
    use_stemming: bool = True
) -> Dict[str, float]:
    """Calculate ROUGE scores for a list of predictions."""
    rouge_types = rouge_types or ["rouge1", "rouge2", "rougeL"]
    rouge_scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=use_stemming)
    scores = {rouge_type: [] for rouge_type in rouge_types}

    for prediction, reference in zip(predictions, references):
        score = rouge_scorer.score(reference, prediction)
        for rouge_type in rouge_types:
            scores[rouge_type].append(score[rouge_type].fmeasure)

    return {rouge_type: np.mean(values) for rouge_type, values in scores.items()}

def calculate_bleu_score(predictions: List[str], references: List[str]) -> float:
    """Calculate BLEU score for a list of predictions."""
    smoothing = SmoothingFunction().method1
    bleu_scores = []

    for prediction, reference in zip(predictions, references):
        prediction_tokens = nltk.word_tokenize(prediction.lower())
        reference_tokens = [nltk.word_tokenize(reference.lower())]
        bleu_scores.append(sentence_bleu(reference_tokens, prediction_tokens, smoothing_function=smoothing))

    return np.mean(bleu_scores)

def calculate_meteor_score(predictions: List[str], references: List[str]) -> float:
    """Calculate METEOR score for a list of predictions."""
    meteor_scores = []

    for prediction, reference in zip(predictions, references):
        meteor_scores.append(meteor_score([reference], prediction))

    return np.mean(meteor_scores)

def calculate_summarization_metrics(
    predictions: List[str],
    references: List[str],
    *,  # Force keyword arguments
    rouge_types: List[str] = None,
    use_stemming: bool = True,
    batch_size: int = 32,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    logger: Optional[logging.Logger] = None
) -> MetricsResult:
    """Calculate all metrics for summarization evaluation."""
    logger = logger or logging.getLogger(__name__)
    rouge_types = rouge_types or ["rouge1", "rouge2", "rougeL"]

    # Calculate ROUGE
    rouge_scores = calculate_rouge_scores(
        predictions,
        references,
        rouge_types=rouge_types,
        use_stemming=use_stemming
    )

    # Calculate BLEU
    bleu = calculate_bleu_score(predictions, references)

    # Calculate METEOR
    meteor = calculate_meteor_score(predictions, references)

    # Calculate means
    mean_scores = {
        "mean_rouge": np.mean(list(rouge_scores.values())),
        "mean_bleu": bleu,
        "mean_meteor": meteor
    }

    return MetricsResult(
        rouge_scores=rouge_scores,
        bleu_score=bleu,
        meteor_score=meteor,
        mean_scores=mean_scores
    )
