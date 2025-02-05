#!/usr/bin/env python3

import argparse
from pathlib import Path
from src.utils.project_logger import ProjectLogger
from src.summarization_pretrained_pipeline import run_complete_pipeline

def parse_args():
    parser = argparse.ArgumentParser(description="Run pretrained summarization pipeline")
    parser.add_argument(
        "--model-name",
        default="facebook/bart-large-cnn",
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--dataset-split",
        choices=["train", "validation", "test"],
        default="validation",
        help="Dataset split to use"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100,
        help="Number of articles to process"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for outputs"
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["cpu", "cuda", "auto"],
        help="Computing device to use"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    logger = ProjectLogger("pretrained_pipeline")
    
    # Ensure output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run pipeline
    results = run_complete_pipeline(
        model_name=args.model_name,
        dataset_split=args.dataset_split,
        sample_size=args.sample_size,
        device=args.device,
        logger=logger
    )
    
    # Save results
    output_file = args.output_dir / f"results_{args.dataset_split}.json"
    logger.save_results(results, output_file)
    
    print(f"\nResults saved to {output_file}")
    print(f"Average ROUGE-1: {results.evaluation_metrics.rouge_scores['rouge1']:.3f}")

if __name__ == "__main__":
    main()
