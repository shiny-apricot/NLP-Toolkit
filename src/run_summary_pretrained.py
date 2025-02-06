"""
Main pipeline script for running pretrained summarization models.
Orchestrates data loading, inference, evaluation, and result saving.
"""

from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import torch
from typing import Optional, List

from src.data_processing.huggingface_data_loader import (
    HuggingFaceLoader,
    TokenizedDataset,
    PreprocessingParams
)
from src.models.abstractive_sum.bart_inference import (
    BartInference,
    InferenceConfig,
    BartConfig
)
from src.evaluation.metrics import (
    calculate_summarization_metrics,
    MetricsResult
)
from src.utils.config_loader import (
    PipelineConfig,
    load_yaml_config
)
from src.utils.project_logger import get_logger
from src.utils.save_model import save_model

@dataclass
class PipelineResults:
    """Container for pipeline execution results."""
    metrics: MetricsResult
    sample_predictions: List[str]
    sample_references: List[str]
    processing_time: float
    dataset_size: int
    model_path: Optional[Path]

def run_summarization_pipeline(
    config_path: Path,
    *,  # Force keyword arguments
    save_model_path: Optional[Path] = None,
    sample_size: Optional[int] = None
) -> PipelineResults:
    """
    Run end-to-end summarization pipeline.

    Args:
        config_path: Path to pipeline configuration YAML
        save_model_path: Optional path to save the model
        sample_size: Optional limit on number of samples to process

    Returns:
        PipelineResults containing metrics and samples
    """
    # Load configuration
    config = load_yaml_config(config_path)
    
    # Initialize logger
    logger = get_logger(
        "summarization_pipeline",
        log_file=config.output_dir / "pipeline.log",
        cloudwatch_group="/summarization/pretrained",
        aws_region="us-west-2"  # Consider moving to config
    )
    
    logger.info(
        "Starting summarization pipeline",
        config_path=str(config_path),
        device=config.device,
        model=config.model_name
    )

    start_time = datetime.now()

    try:
        # Initialize data loader
        loader = HuggingFaceLoader.load_cnn_daily(
            split=config.dataset_split,
            max_samples=sample_size or config.sample_size,
            shuffle=True,
            cache_dir=Path("cache")
        )

        # Load and preprocess dataset
        dataset = loader.load_and_preprocess(
            preprocessing_params=PreprocessingParams(
                remove_html=True,
                normalize_whitespace=True
            )
        )

        # Initialize model for inference
        model_config = BartConfig(
            model_name=config.model_name,
            max_length=config.max_length,
            min_length=config.min_length,
            num_beams=config.num_beams,
            device_map=config.device,
            use_bfloat16=True
        )

        inference_config = InferenceConfig(
            batch_size=8,
            max_length=config.max_length,
            min_length=config.min_length,
            num_return_sequences=1
        )

        # Run inference
        with BartInference(model_config, inference_config) as model:
            predictions = model.summarize_batch(
                dataset.raw_texts,
                return_attention=False
            )
            
            generated_summaries = [p.summary for p in predictions]

            # Calculate metrics
            metrics = calculate_summarization_metrics(
                predictions=generated_summaries,
                references=dataset.raw_summaries,
                device=config.device,
                logger=logger
            )

            # Save model if path provided
            model_path = None
            if save_model_path:
                model_path = save_model(
                    model=model.model,
                    tokenizer=model.tokenizer,
                    save_path=save_model_path,
                    version="1.0.0",
                    description="Pretrained BART model for summarization",
                    performance_metrics=metrics.mean_scores
                )

            # Log results
            logger.info(
                "Pipeline completed successfully",
                rouge_scores=metrics.rouge_scores,
                bleu_score=metrics.bleu_score,
                meteor_score=metrics.meteor_score,
                processing_time=(datetime.now() - start_time).total_seconds(),
                dataset_size=len(dataset.raw_texts)
            )

            # Prepare results
            results = PipelineResults(
                metrics=metrics,
                sample_predictions=generated_summaries[:5],  # First 5 predictions
                sample_references=dataset.raw_summaries[:5],  # First 5 references
                processing_time=(datetime.now() - start_time).total_seconds(),
                dataset_size=len(dataset.raw_texts),
                model_path=model_path
            )

            # Save results
            logger.save_results(
                results,
                config.output_dir / f"results_{datetime.now():%Y%m%d_%H%M%S}.json"
            )

            return results

    except Exception as e:
        logger.error(
            "Pipeline failed",
            error=str(e),
            processing_time=(datetime.now() - start_time).total_seconds()
        )
        raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run summarization pipeline")
    parser.add_argument("--config", type=Path, required=True, help="Path to config YAML")
    parser.add_argument("--save-model", type=Path, help="Path to save model")
    parser.add_argument("--samples", type=int, help="Number of samples to process")
    
    args = parser.parse_args()
    
    results = run_summarization_pipeline(
        config_path=args.config,
        save_model_path=args.save_model,
        sample_size=args.samples
    )
