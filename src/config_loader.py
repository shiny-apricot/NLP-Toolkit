from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import yaml
from pathlib import Path
import os
from functools import lru_cache

class ConfigError(Exception):
    """Base exception for configuration errors."""
    pass

class ConfigValidationError(ConfigError):
    """Raised when configuration validation fails."""
    pass

class ConfigFileNotFoundError(ConfigError):
    """Raised when configuration file is not found."""
    pass

@dataclass
class AWSConfig:
    region: str
    s3_bucket: str
    sagemaker_role_arn: str
    cloudwatch_log_group: str
    cloudwatch_log_stream: str

@dataclass
class LoggingConfig:
    level: str
    log_dir: str
    tensorboard_dir: str

@dataclass
class InstanceConfig:
    instance_type: str
    num_cpus: int
    memory_gb: int
    num_gpus: int
    batch_size: int
    gradient_accumulation_steps: int
    fp16: bool
    num_workers: int
    max_length: int
    model_type: str
    model_name: str
    dtype: str
    cache_dir: str
    aws: AWSConfig
    logging: LoggingConfig

    def __post_init__(self):
        """Validate configuration values after initialization."""
        self._validate_resources()
        self._validate_training_params()
        
    def _validate_resources(self):
        """Validate resource-related configuration."""
        if self.num_cpus < 1:
            raise ConfigValidationError("num_cpus must be at least 1")
        if self.memory_gb < 1:
            raise ConfigValidationError("memory_gb must be at least 1")
        if self.num_gpus < 0:
            raise ConfigValidationError("num_gpus cannot be negative")
            
    def _validate_training_params(self):
        """Validate training-related parameters."""
        if self.batch_size < 1:
            raise ConfigValidationError("batch_size must be at least 1")
        if self.gradient_accumulation_steps < 1:
            raise ConfigValidationError("gradient_accumulation_steps must be at least 1")
        if self.max_length < 1:
            raise ConfigValidationError("max_length must be at least 1")
        if self.num_workers < 0:
            raise ConfigValidationError("num_workers cannot be negative")

    @classmethod
    def from_dict(cls, config: dict) -> 'InstanceConfig':
        return cls(
            instance_type=config["instance_type"],
            num_cpus=config["compute"]["num_cpus"],
            memory_gb=config["compute"]["memory_gb"],
            num_gpus=config["compute"]["num_gpus"],
            batch_size=config["training"]["batch_size"],
            gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
            fp16=config["training"]["fp16"],
            num_workers=config["training"]["num_workers"],
            max_length=config["training"]["max_length"],
            model_type=config["model"]["model_type"],
            model_name=config["model"]["model_name"],
            dtype=config["model"]["dtype"],
            cache_dir=config["training"]["cache_dir"],
            aws=AWSConfig(
                region=config["aws"]["region"],
                s3_bucket=config["aws"]["s3_bucket"],
                sagemaker_role_arn=config["aws"]["sagemaker_role_arn"],
                cloudwatch_log_group=config["aws"]["cloudwatch"]["log_group"],
                cloudwatch_log_stream=config["aws"]["cloudwatch"]["log_stream"]
            ),
            logging=LoggingConfig(
                level=config["logging"]["level"],
                log_dir=config["logging"]["log_dir"],
                tensorboard_dir=config["logging"]["tensorboard_dir"]
            )
        )

class ConfigLoader:
    def __init__(self, config_dir: str = "configs/aws_configs"):
        self.config_dir = Path(config_dir)
        if not self.config_dir.exists():
            raise ConfigError(f"Config directory not found: {config_dir}")
        
    @lru_cache(maxsize=32)
    def load_instance_config(self, instance_type: str) -> InstanceConfig:
        """
        Load and cache instance configuration.
        
        Args:
            instance_type: AWS instance type (e.g., 'p3.2xlarge')
            
        Returns:
            InstanceConfig: Configuration for the specified instance
            
        Raises:
            ConfigFileNotFoundError: If config file doesn't exist
            ConfigValidationError: If config values are invalid
        """
        try:
            config_dict = self._load_config_file(instance_type)
            config_dict = self._apply_env_overrides(config_dict)
            return InstanceConfig.from_dict(config_dict)
        except yaml.YAMLError as e:
            raise ConfigError(f"Error parsing config file: {e}")
        
    def _load_config_file(self, instance_type: str) -> Dict[str, Any]:
        """Load and parse the YAML config file."""
        config_path = self.config_dir / f"{instance_type}.yaml"
        if not config_path.exists():
            raise ConfigFileNotFoundError(
                f"No config found for instance type: {instance_type}"
            )
            
        with open(config_path) as f:
            return yaml.safe_load(f)
            
    def _apply_env_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides to config."""
        env_mappings = {
            "BATCH_SIZE": ("training", "batch_size"),
            "NUM_WORKERS": ("training", "num_workers"),
            "MAX_LENGTH": ("training", "max_length"),
            "MODEL_TYPE": ("model", "model_type"),
            "MODEL_NAME": ("model", "model_name"),
        }
        
        for env_var, (section, key) in env_mappings.items():
            if env_value := os.getenv(env_var):
                config[section][key] = self._convert_env_value(env_value, config[section][key])
                
        return config
        
    def _convert_env_value(self, value: str, original_value: Any) -> Any:
        """Convert environment variable value to the correct type."""
        if isinstance(original_value, bool):
            return value.lower() in ("true", "1", "yes")
        if isinstance(original_value, int):
            return int(value)
        if isinstance(original_value, float):
            return float(value)
        return value
        
    def _create_instance_config(self, config: Dict[str, Any]) -> InstanceConfig:
        """Create InstanceConfig from dictionary."""
        return InstanceConfig(
            instance_type=config["instance_type"],
            num_cpus=config["compute"]["num_cpus"],
            memory_gb=config["compute"]["memory_gb"],
            num_gpus=config["compute"]["num_gpus"],
            batch_size=config["training"]["batch_size"],
            gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
            fp16=config["training"]["fp16"],
            num_workers=config["training"]["num_workers"],
            max_length=config["training"]["max_length"],
            model_type=config["model"]["model_type"],
            model_name=config["model"]["model_name"],
            dtype=config["model"]["dtype"],
            cache_dir=config["training"]["cache_dir"],
            aws=AWSConfig(
                region=config["aws"]["region"],
                s3_bucket=config["aws"]["s3_bucket"],
                sagemaker_role_arn=config["aws"]["sagemaker_role_arn"],
                cloudwatch_log_group=config["aws"]["cloudwatch"]["log_group"],
                cloudwatch_log_stream=config["aws"]["cloudwatch"]["log_stream"]
            ),
            logging=LoggingConfig(
                level=config["logging"]["level"],
                log_dir=config["logging"]["log_dir"],
                tensorboard_dir=config["logging"]["tensorboard_dir"]
            )
        )