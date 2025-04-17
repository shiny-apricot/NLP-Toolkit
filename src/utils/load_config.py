from all_dataclass import Config, DatasetConfig, EvaluationConfig, HardwareConfig, ModelConfig, OutputConfig, TrainingConfig


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
    
    # Handle sample_size of 0 to load all data
    dataset_config = raw_config["dataset"].copy()
    if dataset_config.get("sample_size", 0) == 0:
        logger.info("Sample size is 0, will load all available data.")
        dataset_config["sample_size"] = None
    
    # Create hardware config if present in raw_config
    hardware_config = None
    if "hardware" in raw_config:
        hardware_config = HardwareConfig(**raw_config["hardware"])
    
    return Config(
        dataset=DatasetConfig(**dataset_config),
        model=ModelConfig(**raw_config["model"]),
        training=TrainingConfig(**raw_config["training"]),
        evaluation=EvaluationConfig(**raw_config["evaluation"]),
        output=OutputConfig(**raw_config["output"]),
        hardware=hardware_config
    )