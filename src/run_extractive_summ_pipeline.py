"""
Extractive summarization pipeline.

This module provides a pipeline for extractive text summarization.
It loads text data, applies extractive summarization techniques,
evaluates the resulting summaries, and saves the evaluation metrics.
"""

from dataclasses import dataclass
from typing import Any, List
import argparse
import os
import time
import yaml
from pathlib import Path

from rouge_score import rouge_scorer
from tqdm import tqdm

from all_dataclass import LoadDatasetResult, Metrics, SaveResultsOutput
from data_processing.load_and_save_dataset import load_and_save_dataset
from model.extractive_summarizer import generate_extractive_summary, ExtractiveSummaryResult
from model.save_evaluation_results import save_evaluation_results
from utils.create_timestamped_output_dir import create_timestamped_output_dir
from utils.load_config import load_config
from utils.project_logger import setup_logger


@dataclass
class ExtractiveEvaluationResult:
    """Results from evaluating extractive summarization."""
    metrics: Metrics
    processing_time: float
    summary_examples: List[ExtractiveSummaryResult]


@dataclass
class ExtractivePipelineResult:
    """Complete results from running the extractive summarization pipeline."""
    evaluation_result: ExtractiveEvaluationResult
    save_results: SaveResultsOutput


def evaluate_extractive_summaries(
    summaries: List[ExtractiveSummaryResult],
    references: List[str],
    logger: Any
) -> Metrics:
    """Evaluate extractive summaries using ROUGE metrics.
    
    Args:
        summaries: List of extractive summary results
        references: List of reference summaries
        logger: Logger instance for logging
        
    Returns:
        Metrics: Calculated ROUGE metrics
    """
    logger.info("Evaluating extractive summaries with ROUGE metrics.")
    
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    
    rouge1_total, rouge2_total, rougeL_total = 0.0, 0.0, 0.0
    
    for summary_result, reference in zip(summaries, references):
        summary = summary_result.summary
        scores = scorer.score(reference, summary)
        rouge1_total += scores["rouge1"].fmeasure
        rouge2_total += scores["rouge2"].fmeasure
        rougeL_total += scores["rougeL"].fmeasure
    
    num_samples = len(summaries)
    logger.info(f"Evaluation completed on {num_samples} samples.")
    
    return Metrics(
        rouge_1=rouge1_total / num_samples,
        rouge_2=rouge2_total / num_samples,
        rouge_L=rougeL_total / num_samples
    )


def run_extractive_pipeline(
    config_path: str,
    output_dir: str,
    logger: Any
) -> ExtractivePipelineResult:
    """Run the extractive text summarization pipeline.
    
    Args:
        config_path: Path to configuration file
        output_dir: Directory to store outputs
        logger: Logger instance for logging
        
    Returns:
        ExtractivePipelineResult: Results from the extractive summarization pipeline
    """
    logger.info("Running extractive summarization pipeline.")
    config = load_config(config_path=config_path, logger=logger)
    
    # Load dataset
    loaded_dataset: LoadDatasetResult = load_and_save_dataset(
        sample_size=config.dataset.sample_size,
        dataset=config.dataset,
        logger=logger
    )
    
    # Extract text and reference summaries
    input_texts = []
    reference_summaries = []
    
    # Using test dataset for evaluation
    for item in loaded_dataset.test_dataset:
        input_texts.append(item[config.dataset.input_column])
        reference_summaries.append(item[config.dataset.target_column])
    
    # Limit to a smaller number if needed
    max_samples = min(len(input_texts), 100)  # Process up to 100 samples
    input_texts = input_texts[:max_samples]
    reference_summaries = reference_summaries[:max_samples]
    
    logger.info(f"Processing {len(input_texts)} samples for extractive summarization.")
    
    # Generate extractive summaries
    start_time = time.time()
    summary_results = []
    
    for text in tqdm(input_texts, desc="Generating extractive summaries"):
        summary_result = generate_extractive_summary(
            text=text,
            num_sentences=3  # Can be configurable
        )
        summary_results.append(summary_result)
    
    total_processing_time = time.time() - start_time
    avg_processing_time = total_processing_time / len(input_texts)
    logger.info(f"Extractive summarization completed in {total_processing_time:.2f} seconds.")
    logger.info(f"Average processing time per document: {avg_processing_time:.4f} seconds.")
    
    # Evaluate summaries
    metrics = evaluate_extractive_summaries(
        summaries=summary_results,
        references=reference_summaries,
        logger=logger
    )
    
    # Create evaluation result
    evaluation_result = ExtractiveEvaluationResult(
        metrics=metrics,
        processing_time=total_processing_time,
        summary_examples=summary_results[:5]  # Include a few examples
    )
    
    # Save evaluation results
    save_results = save_evaluation_results(
        metrics=metrics,
        output_dir=output_dir,
        logger=logger
    )
    
    return ExtractivePipelineResult(
        evaluation_result=evaluation_result,
        save_results=save_results
    )


if __name__ == "__main__":
    """Main entry point for the extractive summarization pipeline.
    
    Sets up logging, executes the extractive summarization pipeline,
    and reports success or failure.
    """
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description='Run extractive text summarization pipeline')
    parser.add_argument('--config', type=str, default='test_config.yaml',
                        help='Configuration filename (stored in ./configs/ directory)')
    args = parser.parse_args()
    
    print("Starting extractive text summarization pipeline...")
    
    # Construct full config path using configs directory with Path
    config_path = Path('./configs') / args.config
    
    # Load base config to get output directory
    with open(config_path, "r") as file:
        raw_config = yaml.safe_load(file)
    
    # Create timestamped output directory
    base_output_dir = raw_config["output"]["output_dir"]
    timestamped_dir = create_timestamped_output_dir(base_output_dir)
    print(f"Created output directory: {timestamped_dir}")
    
    # Setup logger with the timestamped directory
    logger = setup_logger('extractive_summarization.log', output_dir=timestamped_dir)
    logger.info(f"Extractive summarization pipeline initialized. Outputs will be saved to: {timestamped_dir}")
    
    try:
        result = run_extractive_pipeline(
            config_path=str(config_path), 
            output_dir=timestamped_dir,
            logger=logger
        )
        logger.info(f"Pipeline completed successfully with ROUGE-L score of {result.evaluation_result.metrics.rouge_L:.4f}")
        if result.save_results.success:
            logger.info(f"Results saved to {result.save_results.file_path}")
        else:
            logger.warning("Pipeline completed but results could not be saved")
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}")
        print(f"Pipeline failed. See logs for details.")
        raise
    else:
        print(f"Extractive summarization pipeline completed successfully. Outputs saved to: {timestamped_dir}")
