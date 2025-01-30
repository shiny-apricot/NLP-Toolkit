from dataclasses import dataclass
from typing import List, Dict, Union, Optional
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
class MetricsConfig:
    """Configuration for evaluation metrics."""
    rouge_types: List[str] = None
    use_stemming: bool = True
    batch_size: int = 32
    num_workers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __post_init__(self):
        if self.rouge_types is None:
            self.rouge_types = ["rouge1", "rouge2", "rougeL"]

class SummarizationMetrics:
    """Calculate various metrics for summarization evaluation."""
    
    def __init__(
        self,
        config: MetricsConfig,
        logger: Optional[logging.Logger] = None
    ):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(
            self.config.rouge_types,
            use_stemmer=config.use_stemming
        )
        
        # Download required NLTK data
        try:
            nltk.download('punkt')
            nltk.download('wordnet')
            nltk.download('omw-1.4')
        except Exception as e:
            self.logger.warning(f"Failed to download NLTK data: {str(e)}")

    def calculate_rouge(
        self,
        prediction: str,
        reference: str
    ) -> Dict[str, float]:
        """Calculate ROUGE scores for a single prediction."""
        try:
            scores = self.rouge_scorer.score(reference, prediction)
            return {
                key: value.fmeasure
                for key, value in scores.items()
            }
        except Exception as e:
            self.logger.error(f"ROUGE calculation failed: {str(e)}")
            return {key: 0.0 for key in self.config.rouge_types}

    def calculate_bleu(
        self,
        prediction: str,
        reference: str
    ) -> float:
        """Calculate BLEU score for a single prediction."""
        try:
            smoothing = SmoothingFunction().method1
            prediction_tokens = nltk.word_tokenize(prediction.lower())
            reference_tokens = [nltk.word_tokenize(reference.lower())]
            return sentence_bleu(reference_tokens, prediction_tokens, smoothing_function=smoothing)
        except Exception as e:
            self.logger.error(f"BLEU calculation failed: {str(e)}")
            return 0.0

    def calculate_meteor(
        self,
        prediction: str,
        reference: str
    ) -> float:
        """Calculate METEOR score for a single prediction."""
        try:
            return meteor_score([reference], prediction)
        except Exception as e:
            self.logger.error(f"METEOR calculation failed: {str(e)}")
            return 0.0

    def calculate_metrics(
        self,
        prediction: str,
        reference: str
    ) -> Dict[str, float]:
        """Calculate all metrics for a single prediction."""
        metrics = {}
        
        # Calculate ROUGE scores
        rouge_scores = self.calculate_rouge(prediction, reference)
        metrics.update(rouge_scores)
        
        # Calculate BLEU score
        metrics['bleu'] = self.calculate_bleu(prediction, reference)
        
        # Calculate METEOR score
        metrics['meteor'] = self.calculate_meteor(prediction, reference)
        
        return metrics

    def batch_calculate_metrics(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, List[float]]:
        """Calculate metrics for a batch of predictions."""
        if len(predictions) != len(references):
            raise ValueError("Number of predictions and references must match")

        batch_metrics: Dict[str, List[float]] = {
            metric: [] for metric in self.config.rouge_types + ['bleu', 'meteor']
        }

        # Process in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.config.num_workers) as executor:
            future_metrics = [
                executor.submit(self.calculate_metrics, pred, ref)
                for pred, ref in zip(predictions, references)
            ]
            
            # Collect results with progress bar
            for future in tqdm(future_metrics, desc="Calculating metrics"):
                metrics = future.result()
                for metric_name, score in metrics.items():
                    batch_metrics[metric_name].append(score)

        # Calculate mean scores
        mean_metrics = {
            f"mean_{key}": np.mean(values)
            for key, values in batch_metrics.items()
        }
        
        # Calculate standard deviation
        std_metrics = {
            f"std_{key}": np.std(values)
            for key, values in batch_metrics.items()
        }
        
        return {
            "individual": batch_metrics,
            "mean": mean_metrics,
            "std": std_metrics
        }

    def __call__(
        self,
        predictions: Union[str, List[str]],
        references: Union[str, List[str]]
    ) -> Dict[str, Union[float, List[float]]]:
        """Calculate metrics for single or batch predictions."""
        if isinstance(predictions, str) and isinstance(references, str):
            return self.calculate_metrics(predictions, references)
        elif isinstance(predictions, list) and isinstance(references, list):
            return self.batch_calculate_metrics(predictions, references)
        else:
            raise ValueError("Predictions and references must both be strings or both be lists")
