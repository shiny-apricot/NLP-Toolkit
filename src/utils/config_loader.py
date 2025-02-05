"""Configuration loader with inheritance support."""

from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from dataclasses import dataclass


@dataclass
class ComputeConfig:
    num_cpus: int
    memory_gb: int
    num_gpus: int
    cuda_visible_devices: str
    torch_num_threads: int

@dataclass
class TrainingConfig:
    batch_size: int
    gradient_accumulation_steps: int
    fp16: bool
    num_workers: int
    max_length: int
    cache_dir: str

@dataclass
class ModelConfig:
    model_type: str
    model_name: str
    dtype: str

@dataclass
class AWSConfig:
    region: str
    s3_bucket: str
    sagemaker_role_arn: str
    cloudwatch: dict

@dataclass
class InstanceConfig:
    instance_type: str
    compute: ComputeConfig
    training: TrainingConfig
    model: ModelConfig
    aws: AWSConfig
    logging: dict
    monitoring: dict
    security: dict
    development: dict

@dataclass
class PipelineConfig:
    """Configuration for the summarization pipeline."""
    model_name: str
    dataset_split: str
    sample_size: int
    device: str
    output_dir: Path
    max_length: int = 512
    min_length: int = 50
    num_beams: int = 4
    instance_type: str = "local"
    compute: Dict[str, Any] = None


def deep_merge(dict1: Dict, dict2: Dict) -> Dict:
    """Deep merge two dictionaries."""
    result = dict1.copy()
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_instance_config(config_path: Path) -> InstanceConfig:
    """Load and validate instance configuration from YAML file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    
    compute_config = ComputeConfig(**config_dict.pop('compute'))
    training_config = TrainingConfig(**config_dict.pop('training'))
    model_config = ModelConfig(**config_dict.pop('model'))
    aws_config = AWSConfig(**config_dict.pop('aws'))
    
    return InstanceConfig(
        instance_type=config_dict.pop('instance_type'),
        compute=compute_config,
        training=training_config,
        model=model_config,
        aws=aws_config,
        logging=config_dict.pop('logging'),
        monitoring=config_dict.pop('monitoring'),
        security=config_dict.pop('security'),
        development=config_dict.pop('development')
    )

def load_yaml_config(config_path: Path) -> PipelineConfig:
    """Load YAML configuration file with inheritance support.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        PipelineConfig object
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
        
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
        
    # Handle inheritance
    if "extends" in config_dict:
        base_path = config_path.parent / config_dict.pop("extends")
        with open(base_path) as f:
            base_config = yaml.safe_load(f)
        config_dict = deep_merge(base_config, config_dict)
    
    # Map the nested config to flat PipelineConfig
    pipeline_config = PipelineConfig(
        model_name=config_dict["model"]["model_name"],
        dataset_split=config_dict["dataset"]["split"],
        sample_size=config_dict["dataset"]["sample_size"],
        device="cuda" if config_dict.get("compute", {}).get("num_gpus", 0) > 0 else "cpu",
        output_dir=Path(config_dict.get("logging", {}).get("log_dir", "outputs")),
        max_length=config_dict["model"].get("max_length", 512),
        min_length=config_dict["model"].get("min_length", 50),
        num_beams=config_dict["model"].get("num_beams", 4),
        instance_type=config_dict.get("instance_type", "local"),
        compute=config_dict.get("compute", {})
    )
    
    return pipeline_config
