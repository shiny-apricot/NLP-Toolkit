"""Text Summarization Pipeline using Pre-trained Models"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Union, Optional, Literal
import numpy as np
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Local imports
from .utils.project_logger import ProjectLogger
from .utils.gpu_manager import GPUManager, GPUConfig
from .models.inference.inference_manager import InferenceManager
from .data.cnn_daily_dataset import load_cnn_daily_dataset
from .validation.dataset_validator import ValidationMetrics, validate_dataset_split
from .evaluation.summary_metrics import EvaluationMetrics, calculate_metrics
from .utils.config_loader import load_yaml_config, PipelineConfig


@dataclass
class SummarizationResult:
    """The result of summarizing a text."""
    summary_text: str
    original_text_length: int
    summary_length: int
    processing_duration_ms: float
    input_token_count: int

    @property
    def compression_ratio(self) -> float:
        """Calculate the compression ratio of the summary."""
        return self.summary_length / self.original_text_length if self.original_text_length > 0 else 0.0


@dataclass
class PipelineResults:
    """Results from running the complete pipeline."""
    validation_metrics: ValidationMetrics
    evaluation_metrics: EvaluationMetrics
    sample_predictions: List[Tuple[str, str, str]]  # [(article, reference, prediction)]
    processing_info: dict


@dataclass
class DatasetSplits:
    """Container for dataset splits."""
    train: Optional[List] = None
    validation: Optional[List] = None
    test: Optional[List] = None
    split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15)


def generate_summary(
    input_text: Union[str, List[str]],
    model_name: str,
    *,
    max_length: int = 512,
    min_length: int = 50,
    num_beams: int = 4,
    length_penalty: float = 1.0,
    early_stopping: bool = True,
    batch_size: int = 8,
    device: str = "auto",
    logger: ProjectLogger
) -> Union[SummarizationResult, List[SummarizationResult]]:
    """Create a summary of the input text(s) using the specified model.
    
    Args:
        input_text: Single text string or list of texts to summarize
        model_name: Name of the pre-trained model to use
        max_length: Maximum length of generated summary
        min_length: Minimum length of generated summary
        num_beams: Number of beams for beam search
        length_penalty: Length penalty for generation
        early_stopping: Whether to stop early in beam search
        batch_size: Number of texts to process at once
        device: Computing device to use ("cpu", "cuda", or "auto")
        logger: Logger instance for tracking

    Returns:
        Single SummarizationResult or list of results
    """
    # Input validation
    if not input_text:
        raise ValueError("Input text cannot be empty")
    if min_length >= max_length:
        raise ValueError("min_length must be less than max_length")

    # Convert input to list
    is_single = isinstance(input_text, str)
    texts = [input_text] if is_single else input_text

    # Setup device and load model
    device = setup_compute_device(device)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)  # Fixed class name
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Initialize inference manager
    inference_mgr = InferenceManager(model, tokenizer, device)

    try:
        # Process all texts in batches
        inference_results = inference_mgr.process_batch(
            texts,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            length_penalty=length_penalty,
            early_stopping=early_stopping
        )

        # Convert to SummarizationResults
        results = [
            SummarizationResult(
                summary_text=result.output_text,
                original_text_length=result.input_length,
                summary_length=result.output_length,
                processing_duration_ms=result.processing_duration_ms,
                input_token_count=result.input_token_count
            )
            for result in inference_results
        ]

        logger.info(
            "Summarization completed",
            processed=len(results),
            avg_compression=sum(r.compression_ratio for r in results) / len(results)
        )

        return results[0] if is_single else results

    except Exception as e:
        logger.error(f"Summarization failed: {str(e)}")
        raise
    finally:
        # Clean up CUDA memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_complete_pipeline(
    model_name: str,
    *,
    dataset_split: Literal["train", "validation", "test"] = "validation",
    split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    sample_size: Optional[int] = 100,
    max_length: int = 512,
    min_length: int = 50,
    num_beams: int = 4,
    device: str = "auto",
    logger: ProjectLogger
) -> PipelineResults:
    """Run complete summarization pipeline including dataset loading, validation, and evaluation.
    
    Args:
        model_name: Name of the pre-trained model to use
        dataset_split: Which dataset split to use ("train", "validation", or "test")
        split_ratios: Tuple of (train, validation, test) ratios that sum to 1.0
        sample_size: Number of articles to process (None for all)
        max_length: Maximum length of generated summary
        min_length: Minimum length of generated summary
        num_beams: Number of beams for beam search
        device: Computing device to use
        logger: Logger instance
    """
    # Validate inputs
    if not model_name:
        raise ValueError("Model name must be provided")
    if sum(split_ratios) != 1.0:
        raise ValueError("Split ratios must sum to 1.0")

    # Load and split dataset
    logger.info("Loading and splitting dataset")
    full_dataset = load_cnn_daily_dataset(split="train")  # Load full dataset
    
    # Randomly shuffle and split dataset
    dataset_size = len(full_dataset.articles)
    indices = np.random.permutation(dataset_size)
    
    train_end = int(dataset_size * split_ratios[0])
    val_end = train_end + int(dataset_size * split_ratios[1])
    
    splits = DatasetSplits(
        train=full_dataset.articles[indices[:train_end]],
        validation=full_dataset.articles[indices[train_end:val_end]],
        test=full_dataset.articles[indices[val_end:]]
    )
    
    # Select appropriate split
    if dataset_split == "train":
        current_split = splits.train
    elif dataset_split == "validation":
        current_split = splits.validation
    else:  # test
        current_split = splits.test
        
    # Apply sample size if specified
    if sample_size:
        current_split = current_split[:sample_size]
    
    # Validate dataset
    logger.info(f"Validating {dataset_split} split")
    validation_metrics = validate_dataset_split(current_split)
    
    # Generate summaries
    logger.info(f"Generating summaries for {dataset_split} split")
    summaries = generate_summary(
        [article.article for article in current_split],
        model_name=model_name,
        max_length=max_length,
        min_length=min_length,
        num_beams=num_beams,
        device=device,
        logger=logger
    )
    
    # Calculate metrics
    logger.info("Calculating evaluation metrics")
    evaluation_metrics = calculate_metrics(
        predictions=[s.summary_text for s in summaries],
        references=[article.highlights for article in current_split],
        processing_times=[s.processing_duration_ms for s in summaries]
    )
    
    # Sample some results for inspection
    sample_idx = min(3, len(summaries))
    sample_predictions = [
        (current_split[i].article,
         current_split[i].highlights,
         summaries[i].summary_text)
        for i in range(sample_idx)
    ]
    
    processing_info = {
        "total_articles": len(current_split),
        "average_processing_time": np.mean([s.processing_duration_ms for s in summaries]),
        "model_name": model_name,
        "device": device
    }
    
    return PipelineResults(
        validation_metrics=validation_metrics,
        evaluation_metrics=evaluation_metrics,
        sample_predictions=sample_predictions,
        processing_info=processing_info
    )

def parse_args():
    parser = argparse.ArgumentParser(description="Run pretrained summarization pipeline")
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to YAML config file"
    )
    # Keep existing arguments as fallback
    parser.add_argument(
        "--model-name",
        default=None,
        help="Override model name from config"
    )
    parser.add_argument(
        "--dataset-split",
        choices=["train", "validation", "test"],
        default=None,
        help="Override dataset split from config"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    logger = ProjectLogger("pretrained_pipeline")
    
    try:
        # Load and validate config
        if args.config:
            config = load_yaml_config(args.config)
            if args.model_name:
                config.model_name = args.model_name
            if args.dataset_split:
                config.dataset_split = args.dataset_split
        else:
            config = PipelineConfig(
                model_name="facebook/bart-large-cnn",
                dataset_split="validation",
                sample_size=100,
                device="auto",
                output_dir=Path("outputs"),
                max_length=512,
                min_length=50,
                num_beams=4
            )
        
        # Validate config
        if not config.model_name:
            raise ValueError("Model name must be provided")
            
        # Ensure output directory exists
        config.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run pipeline with config
        results = run_complete_pipeline(
            model_name=config.model_name,
            dataset_split=config.dataset_split,
            sample_size=config.sample_size,
            device=config.device,
            max_length=config.max_length,
            min_length=config.min_length,
            num_beams=config.num_beams,
            logger=logger
        )
        
        # Save results
        output_file = config.output_dir / f"results_{config.dataset_split}.json"
        logger.save_results(results, output_file)
        
        print(f"\nResults saved to {output_file}")
        print(f"Average ROUGE-1: {results.evaluation_metrics.rouge_scores['rouge1']:.3f}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
