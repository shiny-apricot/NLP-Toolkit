"""
Model saving utility with rich metadata and logging.
Handles both local and S3 storage with comprehensive metadata tracking.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
import json
import shutil
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
import psutil
import platform
import git
from contextlib import contextmanager
import boto3


@dataclass
class HardwareInfo:
    """System hardware information."""
    cpu_count: int = psutil.cpu_count() or 1  # Default to 1 if None is returned
    memory_gb: float = psutil.virtual_memory().total / (1024**3)
    gpu_info: Optional[Dict[str, Any]] = field(default_factory=dict)
    platform_info: str = field(default_factory=lambda: platform.platform())
    python_version: str = field(default_factory=lambda: platform.python_version())

    def __post_init__(self):
        if torch.cuda.is_available():
            self.gpu_info = {
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_count": torch.cuda.device_count(),
                "cuda_version": torch.version.cuda
            }

@dataclass
class GitInfo:
    """Git repository information."""
    commit_hash: str
    branch: str
    uncommitted_changes: bool
    repo_url: Optional[str] = None

    @classmethod
    def from_repo(cls, path: Union[str, Path]) -> 'GitInfo':
        """Create GitInfo from repository path."""
        try:
            repo = git.Repo(path, search_parent_directories=True)
            return cls(
                commit_hash=repo.head.commit.hexsha,
                branch=repo.active_branch.name,
                uncommitted_changes=repo.is_dirty(),
                repo_url=repo.remotes.origin.url if repo.remotes else None
            )
        except git.InvalidGitRepositoryError:
            return cls(
                commit_hash="no_git",
                branch="no_git",
                uncommitted_changes=False
            )

@dataclass
class ModelMetadata:
    """Comprehensive model metadata."""
    model_name: str
    version: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    description: Optional[str] = None
    hardware_info: HardwareInfo = field(default_factory=HardwareInfo)
    git_info: Optional[GitInfo] = None
    training_params: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    file_checksums: Dict[str, str] = field(default_factory=dict)

@dataclass
class SaveConfig:
    """Model saving configuration."""
    path: Path
    s3_bucket: Optional[str] = None
    aws_region: Optional[str] = None
    compress: bool = True
    save_optimizer: bool = True
    include_git_info: bool = True

class ModelSaveError(Exception):
    """Base class for model saving errors."""
    pass

@contextmanager
def save_context(path: Path):
    """Context manager for safe model saving."""
    temp_path = path.parent / f"{path.name}.temp"
    try:
        yield temp_path
        if temp_path.exists():
            if path.exists():
                shutil.rmtree(path)
            temp_path.rename(path)
    finally:
        if temp_path.exists():
            shutil.rmtree(temp_path)

def save_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    save_path: Union[str, Path],
    *,  # Force keyword arguments
    version: str,
    description: Optional[str] = None,
    training_params: Optional[Dict[str, Any]] = None,
    performance_metrics: Optional[Dict[str, Any]] = None,
    s3_bucket: Optional[str] = None,
    aws_region: Optional[str] = None,
    compress: bool = True,
    save_optimizer: bool = True
) -> Path:
    """
    Save model with comprehensive metadata and optional S3 upload.

    Args:
        model: HuggingFace model to save
        tokenizer: Associated tokenizer
        save_path: Local save path
        version: Model version string
        description: Optional model description
        training_params: Optional training parameters
        performance_metrics: Optional model performance metrics
        s3_bucket: Optional S3 bucket for upload
        aws_region: AWS region for S3
        compress: Whether to compress saved files
        save_optimizer: Whether to save optimizer state

    Returns:
        Path where model was saved

    Raises:
        ModelSaveError: If saving fails
    """
    save_path = Path(save_path)
    config = SaveConfig(
        path=save_path,
        s3_bucket=s3_bucket,
        aws_region=aws_region,
        compress=compress,
        save_optimizer=save_optimizer
    )

    try:
        with save_context(save_path) as temp_path:
            # Save model and tokenizer
            model.save_pretrained(
                temp_path,
                save_function=torch.save if not compress else None
            )
            tokenizer.save_pretrained(temp_path)

            # Create metadata
            metadata = ModelMetadata(
                model_name=model.config.name_or_path,
                version=version,
                description=description,
                git_info=GitInfo.from_repo(Path.cwd()) if config.include_git_info else None,
                training_params=training_params or {},
                performance_metrics=performance_metrics or {}
            )

            # Calculate file checksums
            checksums = {}
            for file_path in temp_path.rglob("*"):
                if file_path.is_file():
                    checksums[str(file_path.relative_to(temp_path))] = (
                        hash_file(file_path)
                    )
            metadata.file_checksums = checksums

            # Save metadata
            metadata_path = temp_path / "metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata.__dict__, f, indent=2)

            # Upload to S3 if configured
            if config.s3_bucket and config.aws_region:
                upload_to_s3(temp_path, config)

        return save_path

    except Exception as e:
        raise ModelSaveError(f"Failed to save model: {e}")

def hash_file(path: Path) -> str:
    """Calculate SHA-256 hash of file."""
    import hashlib
    sha256_hash = hashlib.sha256()
    with open(path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def upload_to_s3(path: Path, config: SaveConfig) -> None:
    """Upload saved model to S3."""
    s3 = boto3.client('s3', region_name=config.aws_region)
    
    for file_path in path.rglob("*"):
        if file_path.is_file():
            s3_key = f"models/{path.name}/{file_path.relative_to(path)}"
            s3.upload_file(str(file_path), config.s3_bucket, s3_key)

def load_metadata(path: Union[str, Path]) -> ModelMetadata:
    """Load model metadata from saved path."""
    path = Path(path)
    metadata_path = path / "metadata.json"
    
    if not metadata_path.exists():
        raise ModelSaveError(f"No metadata found at {metadata_path}")
        
    with open(metadata_path) as f:
        data = json.load(f)
        return ModelMetadata(**data)

def verify_checksums(path: Union[str, Path]) -> bool:
    """Verify integrity of saved model files."""
    path = Path(path)
    metadata = load_metadata(path)
    
    for file_path, expected_hash in metadata.file_checksums.items():
        full_path = path / file_path
        if not full_path.exists():
            return False
        if hash_file(full_path) != expected_hash:
            return False
            
    return True