from all_dataclass import Metrics, SaveResultsOutput


import json
import os
from datetime import datetime
from typing import Any


def save_evaluation_results(
    metrics: Metrics,
    output_dir: str,
    logger: Any
) -> SaveResultsOutput:
    """Save evaluation metrics to a JSON file.

    Creates a JSON file containing ROUGE metrics from the evaluation.

    Args:
        metrics: The evaluation metrics (rouge_1, rouge_2, rouge_L)
        output_dir: Directory to save the results
        logger: Logger instance for logging operations

    Returns:
        SaveResultsOutput: Dataclass containing:
            - file_path: Path where results were saved (empty if failed)
            - success: Boolean indicating if the save operation succeeded
    """
    logger.info(f"Saving evaluation results to directory: {output_dir}")
    logger.info(f"Metrics summary - ROUGE-1: {metrics.rouge_1:.4f}, ROUGE-2: {metrics.rouge_2:.4f}, ROUGE-L: {metrics.rouge_L:.4f}")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    file_name = f"evaluation_results.json"
    file_path = os.path.join(output_dir, file_name)

    try:
        # Convert metrics to dictionary
        metrics_dict = {
            "rouge_1": metrics.rouge_1,
            "rouge_2": metrics.rouge_2,
            "rouge_L": metrics.rouge_L,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # Save as JSON
        with open(file_path, 'w') as f:
            json.dump(metrics_dict, f, indent=4)

        logger.info(f"Evaluation results successfully saved to: {file_path}")
        return SaveResultsOutput(file_path=file_path, success=True)
    except IOError as e:
        logger.error(f"I/O error when saving evaluation results: {str(e)}")
        return SaveResultsOutput(file_path="", success=False)
    except Exception as e:
        logger.error(f"Unexpected error when saving evaluation results: {str(e)}")
        return SaveResultsOutput(file_path="", success=False)