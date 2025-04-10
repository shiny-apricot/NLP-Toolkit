from all_dataclass import DatasetConfig, PreprocessedDataset


from typing import Any


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