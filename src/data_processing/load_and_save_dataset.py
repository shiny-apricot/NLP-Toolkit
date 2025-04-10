from all_dataclass import DatasetConfig, LoadDatasetResult


from datasets import load_dataset


import os
from typing import Any


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