"""Main module for orchestrating the text summarization pipeline.

This module defines the high-level workflow for the text summarization project.
"""

from dataclasses import dataclass
import yaml
from typing import Any, Dict, List, Optional
from utils.project_logger import get_logger
from all_dataclass import (
    Config, PipelineResult, DatasetConfig, ModelConfig, TrainingConfig, 
    EvaluationConfig, OutputConfig, LoadDatasetResult, PreprocessedDataset, 
    TrainModelResult, Metrics, EvaluationResult
)
from datasets import load_dataset # type: ignore
import os
from transformers import BartTokenizer, AutoTokenizer, BartForConditionalGeneration, TrainingArguments, Trainer  # type: ignore
from time import time
from rouge_score import rouge_scorer  # type: ignore


def load_config(config_path: str, logger: Any) -> Config:
    """Load the configuration file into a Config dataclass.

    Args:
        config_path: Path to the YAML configuration file.
        logger: Logger instance for logging.

    Returns:
        Config: Parsed configuration as a Config dataclass.
    """
    logger.info(f"Loading configuration from {config_path}.")
    with open(config_path, "r") as file:
        raw_config = yaml.safe_load(file)
    return Config(
        dataset=DatasetConfig(**raw_config["dataset"]),
        model=ModelConfig(**raw_config["model"]),
        training=TrainingConfig(**raw_config["training"]),
        evaluation=EvaluationConfig(**raw_config["evaluation"]),
        output=OutputConfig(**raw_config["output"]),
    )

def run_summarization_pipeline(logger: Any) -> PipelineResult:
    """Run the full text summarization pipeline.

    Args:
        logger: Logger instance for logging.

    Returns:
        PipelineResult: The result of the summarization pipeline, including
        the trained model path and evaluation metrics.
    """
    logger.info("Running summarization pipeline.")
    config = load_config(config_path="./configs/test_config.yaml", logger=logger)

    loaded_dataset: LoadDatasetResult = load_and_save_dataset(
        sample_size=config.dataset.sample_size,
        dataset=config.dataset,
        logger=logger
    )
    tokenizer = load_tokenizer(
        tokenizer_name=config.model.tokenizer_name,
        logger=logger
    )
    preprocessed_dataset: PreprocessedDataset = preprocess_dataset(
        raw_dataset=loaded_dataset.dataset,
        dataset_config=config.dataset,
        tokenizer=tokenizer,
        logger=logger
    )
    # pretrained_cnn_model = get_pretrained_model()
    train_model_result: TrainModelResult = train_model(
        dataset=preprocessed_dataset,
        model_type=config.model.model_type,
        logger=logger
    )
    evaluation_result: Metrics = evaluate_model(
        train_model_result=train_model_result,
        dataset=preprocessed_dataset,
        tokenizer=tokenizer,
        logger=logger
    )
    return PipelineResult(
        train_model_result=train_model_result,
        evaluation_metrics=evaluation_result
    )

def load_and_save_dataset(sample_size: int, dataset: DatasetConfig, logger: Any) -> LoadDatasetResult:
    """Load and preprocess the dataset from Hugging Face and save it locally.

    Args:
        sample_size: Number of samples to load.
        dataset: Dataset configuration.
        logger: Logger instance for logging.

    Returns:
        LoadDatasetResult: Dataclass containing dataset and tokenizer.
    """
    logger.info(f"Loading dataset {dataset.dataset_name} with sample size {sample_size}.")
    
    data_dir = os.path.join(os.path.dirname(__file__), "../data")
    os.makedirs(data_dir, exist_ok=True)
    dataset_path = os.path.join(data_dir, f"{dataset.dataset_name}.arrow")

    hf_dataset = load_dataset(dataset.dataset_name, dataset.dataset_version)

    # take specific sample size
    if sample_size > 0:
        # Sample the dataset for each split
        train_sample_size = min(sample_size, len(hf_dataset["train"]))  # type: ignore
        test_sample_size = min(sample_size, len(hf_dataset["test"]))  # type: ignore
        val_sample_size = min(sample_size, len(hf_dataset["validation"]))  # type: ignore
        
        # Take random samples from each split
        hf_dataset["train"] = hf_dataset["train"].shuffle(seed=42).select(range(train_sample_size))  # type: ignore
        hf_dataset["test"] = hf_dataset["test"].shuffle(seed=42).select(range(test_sample_size))  # type: ignore
        hf_dataset["validation"] = hf_dataset["validation"].shuffle(seed=42).select(range(val_sample_size))  # type: ignore

    result = LoadDatasetResult(
        dataset=hf_dataset,
        train_dataset=hf_dataset["train"].select(range(sample_size)),  # type: ignore
        test_dataset=hf_dataset["test"].select(range(sample_size)),  # type: ignore
        val_dataset=hf_dataset["validation"].select(range(sample_size))  # type: ignore
    )
    return result

def preprocess_dataset(
    raw_dataset: DatasetConfig,
    dataset_config: DatasetConfig,
    tokenizer: Any,
    logger: Any
) -> PreprocessedDataset:
    """Preprocess the raw dataset for training.

    Args:
        raw_dataset: The raw dataset to preprocess.
        tokenizer: The tokenizer to use for preprocessing.
        logger: Logger instance for logging.

    Returns:
        PreprocessedDataset: Dataclass containing preprocessed datasets.
    """
    logger.info("Preprocessing dataset.")
    # Example preprocessing logic
    train_dataset = raw_dataset["train"].map(lambda x: tokenizer(x[dataset_config.input_column])) # type: ignore
    test_dataset = raw_dataset["test"].map(lambda x: tokenizer(x[dataset_config.input_column])) # type: ignore
    val_dataset = raw_dataset["validation"].map(lambda x: tokenizer(x[dataset_config.input_column])) # type: ignore
    logger.info("Dataset preprocessing completed.")

    return PreprocessedDataset(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        val_dataset=val_dataset
    )

def load_tokenizer(tokenizer_name: str, logger: Any) -> Any:
    """Load the tokenizer for the specified model type.

    Args:
        model_type: The type of model (e.g., 'bart').
        logger: Logger instance for logging.

    Returns:
        Any: The loaded tokenizer.
    """
    logger.info(f"Loading tokenizer for tokenizer: {tokenizer_name}.")
    # Example tokenizer loading logic
    return AutoTokenizer.from_pretrained("facebook/bart-large-cnn")


def train_model(
    dataset: PreprocessedDataset,
    model_type: str,
    logger: Any
) -> TrainModelResult:
    """Train the summarization model.

    Args:
        dataset: The preprocessed dataset to use for training.
        model_type: The type of model to train (e.g., 'bart').
        logger: Logger instance for logging.

    Returns:
        TrainModelResult: Dataclass containing the trained model and related info.
    """
    logger.info(f"Initializing model of type {model_type}.")
    model = BartForConditionalGeneration.from_pretrained(model_type)

    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=2,
        logging_dir="./logs",
        logging_steps=10,
    )

    logger.info("Starting model training.")
    start_time = time()
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset.train_dataset,
        eval_dataset=dataset.val_dataset,
    )
    trainer.train()
    training_time = time() - start_time
    logger.info(f"Model training completed in {training_time:.2f} seconds.")

    return TrainModelResult(
        model=model,
        training_args=training_args
    )

def evaluate_model(
    train_model_result: TrainModelResult,
    dataset: PreprocessedDataset,
    tokenizer: Any,
    logger: Any
) -> Metrics:
    """Evaluate the trained model.

    Args:
        train_model_result: Result of the trained model.
        dataset: The dataset to use for evaluation.
        tokenizer: The tokenizer used for decoding.
        logger: Logger instance for logging.

    Returns:
        Metrics: Evaluation metrics.
    """
    logger.info("Evaluating the model.")
    model = train_model_result.model
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    predictions = []
    references = []

    logger.info("Generating predictions for evaluation dataset.")
    for example in dataset.test_dataset:
        inputs = tokenizer(
            example["text"], truncation=True, padding="max_length", return_tensors="pt"
        )
        outputs = model.generate(inputs["input_ids"], max_length=50, num_beams=5)
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predictions.append(prediction)
        references.append(example["summary"])

    logger.info("Calculating ROUGE scores.")
    rouge1, rouge2, rougeL = 0.0, 0.0, 0.0
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        rouge1 += scores["rouge1"].fmeasure
        rouge2 += scores["rouge2"].fmeasure
        rougeL += scores["rougeL"].fmeasure

    num_samples = len(predictions)
    logger.info("Evaluation completed.")
    return Metrics(
        rouge_1=rouge1 / num_samples,
        rouge_2=rouge2 / num_samples,
        rouge_L=rougeL / num_samples
    )

if __name__ == "__main__":
    """Main function to run the text summarization pipeline."""
    print("Starting text summarization pipeline...")
    logger = get_logger(__name__)
    logger.info("Starting text summarization pipeline.")
    result = run_summarization_pipeline(logger=logger)
    logger.info("Pipeline completed successfully.")