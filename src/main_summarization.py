"""Main module for orchestrating the text summarization pipeline.

This module defines the high-level workflow for the text summarization project.
"""

from dataclasses import dataclass
import yaml
from typing import Any, Dict, List, Optional
from utils.project_logger import get_logger, setup_logger
from all_dataclass import (
    Config, PipelineResult, DatasetConfig, ModelConfig, TrainingConfig, 
    EvaluationConfig, OutputConfig, LoadDatasetResult, PreprocessedDataset, 
    TrainModelResult, Metrics, EvaluationResult, SaveResultsOutput
)
from datasets import load_dataset 
import os
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForSeq2SeqLM
from transformers.training_args import TrainingArguments
from transformers.trainer import Trainer
from time import time
from rouge_score import rouge_scorer  
import torch
import json
from datetime import datetime


def create_timestamped_output_dir(base_dir: str) -> str:
    """Create a timestamped directory for outputs.
    
    Args:
        base_dir: Base directory for outputs
        
    Returns:
        Created directory path with timestamp
    """
    # Create timestamp string in format: YYYY-MM-DD_HH-MM-SS
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(base_dir, timestamp)
    
    # Ensure the directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    return output_dir

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
    raw_dataset: Any,
    dataset_config: DatasetConfig,
    tokenizer: Any,
    logger: Any
) -> PreprocessedDataset:
    """Preprocess the raw dataset for training.

    Args:
        raw_dataset: The raw dataset to preprocess.
        dataset_config: Configuration for the dataset.
        tokenizer: The tokenizer to use for preprocessing.
        logger: Logger instance for logging.

    Returns:
        PreprocessedDataset: Dataclass containing preprocessed datasets.
    """
    logger.info("Preprocessing dataset.")
    
    # Use 512 for input length to match model's expected size
    max_input_length = 512
    max_target_length = 128
    
    def process_example(example):
        # Tokenize inputs with padding and truncation
        model_inputs = tokenizer(
            example[dataset_config.input_column],
            max_length=max_input_length,
            padding="max_length",
            truncation=True
        )
        
        # Tokenize targets with padding and truncation
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                example[dataset_config.target_column],
                max_length=max_target_length,
                padding="max_length",
                truncation=True
            )
            
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    # Apply processing to all splits
    train_dataset = raw_dataset["train"].map(
        process_example, 
        batched=True, 
        remove_columns=raw_dataset["train"].column_names
    )
    
    test_dataset = raw_dataset["test"].map(
        process_example, 
        batched=True, 
        remove_columns=raw_dataset["test"].column_names
    )
    
    val_dataset = raw_dataset["validation"].map(
        process_example, 
        batched=True, 
        remove_columns=raw_dataset["validation"].column_names
    )
    
    logger.info("Dataset preprocessing completed.")

    return PreprocessedDataset(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        val_dataset=val_dataset
    )

def load_tokenizer(
    model_name: str, 
    logger: Any
) -> Any:
    """Load the tokenizer for the specified model.

    Args:
        model_name: The name of the model/tokenizer to load.
        logger: Logger instance for logging.

    Returns:
        Any: The loaded tokenizer.
    """
    logger.info(f"Loading tokenizer: {model_name}.")
    return AutoTokenizer.from_pretrained(model_name)

def train_model(
    dataset: PreprocessedDataset,
    model_name: str,
    output_dir: str,
    logger: Any
) -> TrainModelResult:
    """Train the summarization model.

    Args:
        dataset: The preprocessed dataset to use for training.
        model_name: The name of the model to train.
        output_dir: Directory to save model outputs
        logger: Logger instance for logging.

    Returns:
        TrainModelResult: Dataclass containing the trained model and related info.
    """
    logger.info(f"Initializing model: {model_name}.")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=2,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=10,
        report_to="none",  # Disable wandb reporting
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
    for i, example in enumerate(dataset.test_dataset):
        # Convert list to PyTorch tensor
        input_ids = torch.tensor(example["input_ids"]).unsqueeze(0)
        outputs = model.generate(input_ids, max_length=128, num_beams=5)
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Decode reference from label ids
        reference = tokenizer.decode(example["labels"], skip_special_tokens=True)
        
        predictions.append(prediction)
        references.append(reference)
        
        # Limit evaluation to a reasonable number to avoid long processing
        if i >= 100:
            break

    logger.info("Calculating ROUGE scores.")
    rouge1, rouge2, rougeL = 0.0, 0.0, 0.0
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        rouge1 += scores["rouge1"].fmeasure
        rouge2 += scores["rouge2"].fmeasure
        rougeL += scores["rougeL"].fmeasure

    num_samples = len(predictions)
    logger.info(f"Evaluation completed on {num_samples} samples.")
    return Metrics(
        rouge_1=rouge1 / num_samples,
        rouge_2=rouge2 / num_samples,
        rouge_L=rougeL / num_samples
    )

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

if __name__ == "__main__":
    """Main entry point for the text summarization pipeline.
    
    Sets up logging, executes the complete summarization pipeline,
    and reports success or failure.
    """
    print("Starting text summarization pipeline...")
    
    # Load base config to get output directory
    with open("./configs/test_config.yaml", "r") as file:
        raw_config = yaml.safe_load(file)
    
    # Create timestamped output directory
    base_output_dir = raw_config["output"]["output_dir"]
    timestamped_dir = create_timestamped_output_dir(base_output_dir)
    print(f"Created output directory: {timestamped_dir}")
    
    # Setup logger with the timestamped directory
    logger = setup_logger('summarization.log', output_dir=timestamped_dir)
    logger.info(f"Text summarization pipeline initialized. Outputs will be saved to: {timestamped_dir}")
    
    try:
        result = run_summarization_pipeline(
            config_path="./configs/test_config.yaml", 
            output_dir=timestamped_dir,
            logger=logger
        )
        logger.info(f"Pipeline completed successfully with ROUGE-L score of {result.evaluation_metrics.rouge_L:.4f}")
        if result.save_results.success:
            logger.info(f"Results saved to {result.save_results.file_path}")
        else:
            logger.warning("Pipeline completed but results could not be saved")
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}")
        print(f"Pipeline failed. See logs for details.")
        raise
    else:
        print(f"Text summarization pipeline completed successfully. Outputs saved to: {timestamped_dir}")