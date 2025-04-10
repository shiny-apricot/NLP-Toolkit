from all_dataclass import LoadDatasetResult, Metrics, PipelineResult, PreprocessedDataset, TrainModelResult
from data_processing.load_and_save_dataset import load_and_save_dataset
from data_processing.load_tokenizer import load_tokenizer
from data_processing.preprocess_dataset import preprocess_dataset
from model.evaluate_model import evaluate_model
from model.save_evaluation_results import save_evaluation_results
from model.train_model import train_model
from utils.load_config import load_config


import os
from typing import Any


def run_summarization_pipeline(
    config_path: str,
    output_dir: str,
    logger: Any
) -> PipelineResult:
    """Run the full text summarization pipeline.

    Args:
        config_path: Path to configuration file
        output_dir: Directory to store outputs
        logger: Logger instance for logging.

    Returns:
        PipelineResult: The result of the summarization pipeline, including
        the trained model path and evaluation metrics.
    """
    logger.info("Running summarization pipeline.")
    config = load_config(config_path=config_path, logger=logger)

    loaded_dataset: LoadDatasetResult = load_and_save_dataset(
        sample_size=config.dataset.sample_size,
        dataset=config.dataset,
        logger=logger
    )
    tokenizer = load_tokenizer(
        model_name=config.model.model_name,
        logger=logger
    )
    preprocessed_dataset: PreprocessedDataset = preprocess_dataset(
        raw_dataset=loaded_dataset.dataset,
        dataset_config=config.dataset,
        tokenizer=tokenizer,
        logger=logger
    )

    # Use timestamped output directory for model outputs
    model_output_dir = os.path.join(output_dir, "model")
    os.makedirs(model_output_dir, exist_ok=True)

    train_model_result: TrainModelResult = train_model(
        dataset=preprocessed_dataset,
        model_name=config.model.model_name,
        output_dir=model_output_dir,
        logger=logger
    )
    evaluation_result: Metrics = evaluate_model(
        train_model_result=train_model_result,
        dataset=preprocessed_dataset,
        tokenizer=tokenizer,
        logger=logger
    )

    # Save the evaluation results to the timestamped directory
    save_results = save_evaluation_results(
        metrics=evaluation_result,
        output_dir=output_dir,
        logger=logger
    )

    return PipelineResult(
        train_model_result=train_model_result,
        evaluation_metrics=evaluation_result,
        save_results=save_results
    )