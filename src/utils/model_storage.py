"""
Utilities for managing model storage in SageMaker environments.

Provides functions for storing and loading models from either S3 or local storage.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, Literal
import logging
from urllib.parse import urlparse

@dataclass
class StorageLocation:
    """Represents a storage location for models."""
    uri: str
    storage_type: Literal["s3", "local"]
    
    @property
    def is_s3(self) -> bool:
        """Check if storage is S3-based."""
        return self.storage_type == "s3"
    
    @property
    def formatted_uri(self) -> str:
        """Get URI with proper formatting for SageMaker."""
        if self.is_s3:
            return self.uri
        return f"file://{self.uri}" if not self.uri.startswith("file://") else self.uri


def parse_storage_location(
    path: str,
    *,
    logger: Optional[logging.Logger] = None
) -> StorageLocation:
    """
    Parse a path string into a structured storage location.
    
    Args:
        path: Path to parse (can be S3 URI or local path)
        logger: Optional logger for warnings/info
    
    Returns:
        StorageLocation object with parsed information
    """
    if path.startswith("s3://"):
        return StorageLocation(uri=path, storage_type="s3")
        
    # Handle local paths - ensure they're absolute
    local_path = Path(path)
    abs_path = str(local_path.absolute())
    
    if logger:
        logger.info(f"Using local storage path: {abs_path}")
    
    return StorageLocation(uri=abs_path, storage_type="local")


def format_input_path(
    path: str,
    *,
    is_directory: bool = True,
) -> str:
    """
    Format a path for SageMaker input locations.
    
    Args:
        path: Path to format (S3 URI or local path)
        is_directory: Whether the path is a directory
    
    Returns:
        Properly formatted path for SageMaker
    """
    storage_loc = parse_storage_location(path)
    
    # Return S3 paths unchanged
    if storage_loc.is_s3:
        return storage_loc.uri
    
    # Format local paths correctly for SageMaker
    local_path = Path(storage_loc.uri)
    
    # Ensure directory exists
    if is_directory and not local_path.exists():
        local_path.mkdir(parents=True, exist_ok=True)
    
    # Format as file:// URI
    if not storage_loc.uri.startswith("file://"):
        return f"file://{storage_loc.uri}"
    return storage_loc.uri


def get_default_model_location(
    model_name: str,
    *,
    base_dir: Optional[str] = None,
    use_s3: bool = False,
    s3_bucket: Optional[str] = None,
    s3_prefix: str = "models"
) -> str:
    """
    Get the default storage location for a model.
    
    Args:
        model_name: Name of the model
        base_dir: Base directory for local storage
        use_s3: Whether to use S3 storage
        s3_bucket: S3 bucket name (required if use_s3=True)
        s3_prefix: S3 prefix for models folder
    
    Returns:
        Location URI for model
    
    Raises:
        ValueError: If use_s3=True but no bucket specified
    """
    if use_s3:
        if not s3_bucket:
            raise ValueError("s3_bucket must be specified when use_s3=True")
        return f"s3://{s3_bucket}/{s3_prefix}/{model_name}"
    
    # Local storage path
    local_base = base_dir or "/tmp/summarization-models"
    return str(Path(local_base) / model_name)