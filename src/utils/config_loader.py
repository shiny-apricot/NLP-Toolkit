"""Configuration loading and validation utilities."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import yaml

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
