from all_dataclass import DatasetConfig, LoadDatasetResult, DatasetStatistics
import numpy as np

from datasets import load_dataset


import os
from typing import Any, Dict, Optional, List


def compute_dataset_statistics(
    dataset: Any,
    article_field: str,
    summary_field: str
) -> DatasetStatistics:
    """Compute comprehensive statistics about the dataset.
    
    Args:
        dataset: The dataset to analyze
        article_field: The field name for articles/inputs
        summary_field: The field name for summaries/targets
        
    Returns:
        DatasetStatistics dataclass with calculated metrics
    """
    # Calculate length statistics for articles
    article_lengths = [len(item[article_field].split()) for item in dataset]
    article_char_lengths = [len(item[article_field]) for item in dataset]
    
    # Calculate length statistics for summaries
    summary_lengths = [len(item[summary_field].split()) for item in dataset]
    summary_char_lengths = [len(item[summary_field]) for item in dataset]
    
    # Create length distributions (binned)
    article_length_bins = {
        "0-100": 0, "101-250": 0, "251-500": 0, "501-1000": 0, "1000+": 0
    }
    summary_length_bins = {
        "0-25": 0, "26-50": 0, "51-75": 0, "76-100": 0, "100+": 0
    }
    
    # Fill bins for articles
    for length in article_lengths:
        if length <= 100: article_length_bins["0-100"] += 1
        elif length <= 250: article_length_bins["101-250"] += 1
        elif length <= 500: article_length_bins["251-500"] += 1
        elif length <= 1000: article_length_bins["501-1000"] += 1
        else: article_length_bins["1000+"] += 1
    
    # Fill bins for summaries
    for length in summary_lengths:
        if length <= 25: summary_length_bins["0-25"] += 1
        elif length <= 50: summary_length_bins["26-50"] += 1
        elif length <= 75: summary_length_bins["51-75"] += 1
        elif length <= 100: summary_length_bins["76-100"] += 1
        else: summary_length_bins["100+"] += 1
        
    return DatasetStatistics(
        article_max_length=max(article_lengths),
        article_min_length=min(article_lengths),
        article_avg_length=float(np.mean(article_lengths)),
        article_median_length=float(np.median(article_lengths)),
        article_std_length=float(np.std(article_lengths)),
        article_max_char_length=max(article_char_lengths),
        article_avg_char_length=float(np.mean(article_char_lengths)),
        
        summary_max_length=max(summary_lengths),
        summary_min_length=min(summary_lengths),
        summary_avg_length=float(np.mean(summary_lengths)),
        summary_median_length=float(np.median(summary_lengths)),
        summary_std_length=float(np.std(summary_lengths)),
        summary_max_char_length=max(summary_char_lengths),
        summary_avg_char_length=float(np.mean(summary_char_lengths)),
        
        article_length_distribution=article_length_bins,
        summary_length_distribution=summary_length_bins,
        
        dataset_size=len(dataset),
        compression_ratio=float(np.mean([s/a if a > 0 else 0 for s, a in zip(summary_lengths, article_lengths)]))
    )


def load_and_save_dataset(sample_size: Optional[int], dataset: DatasetConfig, logger: Any) -> LoadDatasetResult:
    """Load and preprocess the dataset from Hugging Face and save it locally.

    Args:
        sample_size: Number of samples to load. If None, loads all available data.
        dataset: Dataset configuration.
        logger: Logger instance for logging.

    Returns:
        LoadDatasetResult: Dataclass containing dataset and tokenizer.
    """
    logger.info(f"Loading dataset {dataset.dataset_name} with sample size {sample_size if sample_size is not None else 'ALL'}.")

    data_dir = os.path.join(os.path.dirname(__file__), "../data")
    os.makedirs(data_dir, exist_ok=True)
    dataset_path = os.path.join(data_dir, f"{dataset.dataset_name}.arrow")

    hf_dataset = load_dataset(dataset.dataset_name, dataset.dataset_version)
    
    # Compute statistics on full dataset before sampling
    article_field = dataset.input_column
    summary_field = dataset.target_column
    
    # Get statistics for the full dataset
    full_train_stats = compute_dataset_statistics(hf_dataset["train"], article_field, summary_field) # type: ignore
    full_test_stats = compute_dataset_statistics(hf_dataset["test"], article_field, summary_field) # type: ignore
    full_val_stats = compute_dataset_statistics(hf_dataset["validation"], article_field, summary_field) # type: ignore
    
    logger.info(f"Full dataset statistics computed successfully")
    logger.info(f"Full train set - Articles: max length={full_train_stats.article_max_length}, avg length={full_train_stats.article_avg_length:.2f}")
    logger.info(f"Full train set - Summaries: max length={full_train_stats.summary_max_length}, avg length={full_train_stats.summary_avg_length:.2f}")
    logger.info(f"Full dataset contains {full_train_stats.dataset_size + full_test_stats.dataset_size + full_val_stats.dataset_size} examples")

    # Sample the dataset if sample_size is provided
    if sample_size is not None and sample_size > 0:
        # Sample the dataset for each split
        train_sample_size = min(sample_size, len(hf_dataset["train"]))  # type: ignore
        test_sample_size = min(sample_size, len(hf_dataset["test"]))  # type: ignore
        val_sample_size = min(sample_size, len(hf_dataset["validation"]))  # type: ignore

        # Take random samples from each split
        hf_dataset["train"] = hf_dataset["train"].shuffle(seed=42).select(range(train_sample_size))  # type: ignore
        hf_dataset["test"] = hf_dataset["test"].shuffle(seed=42).select(range(test_sample_size))  # type: ignore
        hf_dataset["validation"] = hf_dataset["validation"].shuffle(seed=42).select(range(val_sample_size))  # type: ignore
        
        train_dataset = hf_dataset["train"]  # type: ignore
        test_dataset = hf_dataset["test"]  # type: ignore
        val_dataset = hf_dataset["validation"]  # type: ignore
        
        # Compute statistics on sampled dataset
        sample_train_stats = compute_dataset_statistics(train_dataset, article_field, summary_field)
        sample_test_stats = compute_dataset_statistics(test_dataset, article_field, summary_field)
        sample_val_stats = compute_dataset_statistics(val_dataset, article_field, summary_field)
        
        logger.info(f"Sampled dataset statistics computed successfully")
        logger.info(f"Sampled train set - Articles: max length={sample_train_stats.article_max_length}, avg length={sample_train_stats.article_avg_length:.2f}")
        logger.info(f"Sampled train set - Summaries: max length={sample_train_stats.summary_max_length}, avg length={sample_train_stats.summary_avg_length:.2f}")
    else:
        # Use all data when sample_size is None
        train_dataset = hf_dataset["train"]  # type: ignore
        test_dataset = hf_dataset["test"]  # type: ignore
        val_dataset = hf_dataset["validation"]  # type: ignore
        
        # When not sampling, the sampled stats are the same as full stats
        sample_train_stats = full_train_stats
        sample_test_stats = full_test_stats
        sample_val_stats = full_val_stats
    
    result = LoadDatasetResult(
        dataset=hf_dataset,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        val_dataset=val_dataset,
        train_statistics=sample_train_stats,
        test_statistics=sample_test_stats,
        validation_statistics=sample_val_stats,
        full_train_statistics=full_train_stats,
        full_test_statistics=full_test_stats,
        full_validation_statistics=full_val_stats
    )
    return result