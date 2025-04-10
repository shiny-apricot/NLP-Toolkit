from all_dataclass import Config, DatasetConfig, EvaluationConfig, ModelConfig, OutputConfig, TrainingConfig


import yaml


from typing import Any


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