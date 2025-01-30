from dataclasses import dataclass
from typing import Optional
import yaml
from pathlib import Path

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

class ConfigLoader:
    def __init__(self, config_dir: str = "configs/instances"):
        self.config_dir = Path(config_dir)
    
    def load_instance_config(self, instance_type: str) -> InstanceConfig:
        config_path = self.config_dir / f"{instance_type}.yaml"
        if not config_path.exists():
            raise ValueError(f"No config found for instance type: {instance_type}")
        
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
            
        return InstanceConfig(
            instance_type=config_dict["instance_type"],
            num_cpus=config_dict["compute"]["num_cpus"],
            memory_gb=config_dict["compute"]["memory_gb"],
            num_gpus=config_dict["compute"]["num_gpus"],
            batch_size=config_dict["training"]["batch_size"],
            gradient_accumulation_steps=config_dict["training"]["gradient_accumulation_steps"],
            fp16=config_dict["training"]["fp16"],
            num_workers=config_dict["training"]["num_workers"],
            max_length=config_dict["training"]["max_length"],
            model_type=config_dict["model"]["model_type"],
            model_name=config_dict["model"]["model_name"],
            dtype=config_dict["model"]["dtype"],
            cache_dir=config_dict["training"]["cache_dir"]
        )