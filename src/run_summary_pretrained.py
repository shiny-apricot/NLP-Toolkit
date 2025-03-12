"""
Main pipeline script for running pretrained summarization models.
Orchestrates data loading, inference, evaluation, and result saving.
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List
import torch
from datetime import datetime

from data_processing.huggingface_data_loader import (
    HuggingFaceLoader,
    DatasetError
)
from models.abstractive_sum.bart_inference import (
    BartInference,
    InferenceConfig,
)
from models.abstractive_sum.bart_abstractive_model import (
    BartConfig,
    SummarizationResult
)
from utils.project_logger import get_logger
from utils.config_loader import load_yaml_config, PipelineConfig
from utils.aws_utils import AWSModelStorage


@dataclass
class PipelineResults:
    """Container for pipeline execution results."""
    summaries: List[SummarizationResult]
    processing_time: float
    input_samples: int
    successful_samples: int
    failed_samples: int
    model_name: str
    dataset_name: str
    timestamp: str = datetime.utcnow().isoformat()


def run_summary_pretrained(
    config_path: Path,
    *,  # Force keyword arguments
    output_dir: Optional[Path] = None,
    aws_bucket: Optional[str] = None,
    aws_region: Optional[str] = None
) -> PipelineResults:
    """
    Run summarization pipeline with pretrained model.

    Args:
        config_path: Path to pipeline configuration file
        output_dir: Directory for saving results
        aws_bucket: Optional S3 bucket for model/results storage
        aws_region: AWS region for S3 bucket

    Returns:
        PipelineResults containing summarization outputs and metrics
    """
    # Initialize logger
    logger = get_logger(
        "summarization_pipeline",
        log_file=Path("logs/pipeline.log"),
        cloudwatch_group="/aws/summarization" if aws_bucket else None,
        aws_region=aws_region
    )
    
    try:
        # Load configuration
        config = load_yaml_config(config_path)
        output_dir = output_dir or Path("outputs")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize AWS storage if specified
        storage = None
        if aws_bucket and aws_region:
            storage = AWSModelStorage(
                bucket=aws_bucket,
                region=aws_region,
                model_prefix="summarization_models",
                logger=logger
            )

        # Initialize model config
        model_config = BartConfig(
            model_name=config.model_name,
            max_length=config.max_length,
            min_length=config.min_length,
            num_beams=config.num_beams,
            device_map="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        inference_config = InferenceConfig(
            batch_size=8,
            max_length=config.max_length,
            min_length=config.min_length
        )

        # Initialize BART model first for its tokenizer
        with BartInference(model_config, inference_config) as model:
            try:
                # Create data loader with model's tokenizer
                loader = HuggingFaceLoader(
                    dataset_config=HuggingFaceLoader.DATASET_CONFIGS["cnn_dailymail"],
                    tokenizer=model.tokenizer,
                    max_length=config.max_length,
                    batch_size=8,
                    num_workers=4
                )

                # Load and preprocess dataset
                processed_dataset = loader.load_and_preprocess(
                    split=config.dataset_split,
                    max_length=config.max_length,
                    truncation=True,
                    padding=True,
                    use_cache=True
                )

                start_time = datetime.now()
                logger.info("Starting summarization pipeline", 
                        model=config.model_name,
                        dataset_split=config.dataset_split,
                        samples=len(processed_dataset.raw_texts))

                # Run inference
                summaries = model.summarize_batch(
                    processed_dataset.raw_texts,
                    return_attention=True
                )

                # Calculate success/failure counts
                successful = len([s for s in summaries if s.summary])
                failed = len(summaries) - successful

                # Create results container
                results = PipelineResults(
                    summaries=summaries,
                    processing_time=(datetime.now() - start_time).total_seconds(),
                    input_samples=len(processed_dataset.raw_texts),
                    successful_samples=successful,
                    failed_samples=failed,
                    model_name=config.model_name,
                    dataset_name=loader.dataset_name
                )

                # Save results
                output_file = output_dir / f"summaries_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                logger.save_results(results, output_file)

                # Save model if using AWS
                if storage:
                    model.save(
                        Path("models/latest"),
                        version="latest",
                        description="Latest inference model",
                        performance_metrics={
                            "successful_samples": successful,
                            "failed_samples": failed,
                            "processing_time": results.processing_time
                        }
                    )

                logger.info("Pipeline completed successfully",
                          processing_time=results.processing_time,
                          successful=successful,
                          failed=failed)

                return results

            except Exception as e:
                logger.error("Pipeline execution failed", error=str(e))
                raise

    except Exception as e:
        logger.error("Pipeline initialization failed", error=str(e))
        raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run summarization pipeline")
    parser.add_argument("--config", type=Path, required=True, help="Path to config file")
    parser.add_argument("--output-dir", type=Path, help="Output directory")
    parser.add_argument("--aws-bucket", help="AWS S3 bucket name")
    parser.add_argument("--aws-region", help="AWS region")
    
    args = parser.parse_args()
    
    results = run_summary_pretrained(
        args.config,
        output_dir=args.output_dir,
        aws_bucket=args.aws_bucket,
        aws_region=args.aws_region
    )