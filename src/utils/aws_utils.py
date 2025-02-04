"""
AWS utilities for model storage and resource management.
Provides S3 integration with local fallback for model storage.
"""

import os
import json
from pathlib import Path
from typing import Optional, Union, Dict, Any
from datetime import datetime
import boto3
from botocore.exceptions import ClientError
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from .project_logger import ProjectLogger, get_logger

class ModelStorageError(Exception):
    """Base exception for model storage operations."""
    pass

class S3UploadError(ModelStorageError):
    """Raised when S3 upload fails."""
    pass

class S3DownloadError(ModelStorageError):
    """Raised when S3 download fails."""
    pass

class AWSModelStorage:
    """Handles model storage operations with S3 integration."""

    def __init__(
        self,
        *,  # Force named parameters
        bucket: str,
        region: str,
        model_prefix: str = "models",
        checkpoint_prefix: str = "checkpoints",
        logger: Optional[ProjectLogger] = None
    ) -> None:
        """
        Initialize AWS model storage.

        Args:
            bucket: S3 bucket name
            region: AWS region
            model_prefix: Prefix for model storage in S3
            checkpoint_prefix: Prefix for checkpoint storage in S3
            logger: Optional logger instance
        """
        self.bucket = bucket
        self.region = region
        self.model_prefix = model_prefix
        self.checkpoint_prefix = checkpoint_prefix
        self.logger = logger or get_logger("aws_storage")
        
        try:
            self.s3 = boto3.client('s3', region_name=region)
            self.s3_available = True
        except Exception as e:
            self.logger.warning(f"S3 initialization failed: {e}. Using local storage.")
            self.s3_available = False

    def _get_s3_path(
        self,
        model_name: str,
        version: str,
        is_checkpoint: bool = False
    ) -> str:
        """Generate S3 path for model or checkpoint."""
        prefix = self.checkpoint_prefix if is_checkpoint else self.model_prefix
        return f"{prefix}/{model_name}/v{version}"

    def save_model(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        model_name: str,
        version: str,
        metadata: Optional[Dict[str, Any]] = None,
        local_path: Optional[Path] = None
    ) -> Path:
        """
        Save model to S3 or local storage.

        Args:
            model: The model to save
            tokenizer: The tokenizer to save
            model_name: Name of the model
            version: Version string
            metadata: Optional metadata about the model
            local_path: Optional local path override

        Returns:
            Path where model was saved

        Raises:
            S3UploadError: If S3 upload fails
        """
        # Create temporary local path if none provided
        local_path = local_path or Path(f"./tmp/models/{model_name}/v{version}")
        local_path.mkdir(parents=True, exist_ok=True)

        # Save model and tokenizer locally
        try:
            model.save_pretrained(local_path)
            tokenizer.save_pretrained(local_path)

            # Save metadata if provided
            if metadata:
                metadata.update({
                    "saved_at": datetime.utcnow().isoformat(),
                    "model_name": model_name,
                    "version": version
                })
                metadata_path = local_path / "metadata.json"
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)

        except Exception as e:
            raise ModelStorageError(f"Failed to save model locally: {e}")

        # Upload to S3 if available
        if self.s3_available:
            try:
                s3_prefix = self._get_s3_path(model_name, version)
                
                # Upload all files in the local directory
                for file_path in local_path.rglob("*"):
                    if file_path.is_file():
                        s3_key = f"{s3_prefix}/{file_path.relative_to(local_path)}"
                        self.s3.upload_file(
                            str(file_path),
                            self.bucket,
                            s3_key
                        )

                self.logger.info(
                    "Model uploaded to S3",
                    extra={
                        "bucket": self.bucket,
                        "prefix": s3_prefix,
                        "model_name": model_name,
                        "version": version
                    }
                )

            except Exception as e:
                raise S3UploadError(f"Failed to upload model to S3: {e}")

        return local_path

    def load_model(
        self,
        model_name: str,
        version: str,
        *,  # Force named parameters
        local_path: Optional[Path] = None,
        device_map: str = "auto",
        use_bfloat16: bool = True
    ) -> tuple[PreTrainedModel, PreTrainedTokenizer, Optional[Dict[str, Any]]]:
        """
        Load model from S3 or local storage.

        Args:
            model_name: Name of the model
            version: Version string
            local_path: Optional local path override
            device_map: Device mapping for model loading
            use_bfloat16: Whether to use bfloat16 precision

        Returns:
            Tuple of (model, tokenizer, metadata)

        Raises:
            S3DownloadError: If S3 download fails
        """
        local_path = local_path or Path(f"./tmp/models/{model_name}/v{version}")

        if self.s3_available and not local_path.exists():
            try:
                s3_prefix = self._get_s3_path(model_name, version)
                local_path.mkdir(parents=True, exist_ok=True)

                # List and download all files under the prefix
                response = self.s3.list_objects_v2(
                    Bucket=self.bucket,
                    Prefix=s3_prefix
                )

                for obj in response.get('Contents', []):
                    file_key = obj['Key']
                    local_file = local_path / Path(file_key).relative_to(s3_prefix)
                    local_file.parent.mkdir(parents=True, exist_ok=True)
                    
                    self.s3.download_file(
                        self.bucket,
                        file_key,
                        str(local_file)
                    )

            except Exception as e:
                raise S3DownloadError(f"Failed to download model from S3: {e}")

        try:
            dtype = torch.bfloat16 if use_bfloat16 else torch.float32
            model = PreTrainedModel.from_pretrained(
                local_path,
                device_map=device_map,
                torch_dtype=dtype
            )
            tokenizer = PreTrainedTokenizer.from_pretrained(local_path)

            metadata = None
            metadata_path = local_path / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path) as f:
                    metadata = json.load(f)

            return model, tokenizer, metadata

        except Exception as e:
            raise ModelStorageError(f"Failed to load model: {e}")

    def save_checkpoint(
        self,
        state_dict: Dict[str, Any],
        *,  # Force named parameters
        model_name: str,
        checkpoint_name: str,
        local_dir: Optional[Path] = None
    ) -> None:
        """
        Save training checkpoint to S3 or locally.

        Args:
            state_dict: Training state to save
            model_name: Name of the model
            checkpoint_name: Name of the checkpoint
            local_dir: Optional local directory override
        """
        local_path = (local_dir or Path("./tmp/checkpoints")) / model_name / f"{checkpoint_name}.pt"
        local_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(state_dict, local_path)

        if self.s3_available:
            try:
                s3_key = f"{self.checkpoint_prefix}/{model_name}/{checkpoint_name}.pt"
                self.s3.upload_file(
                    str(local_path),
                    self.bucket,
                    s3_key
                )
            except Exception as e:
                self.logger.warning(f"Failed to upload checkpoint to S3: {e}")

    def load_checkpoint(
        self,
        model_name: str,
        checkpoint_name: str
    ) -> Optional[Dict[str, Any]]:
        """
        Load checkpoint from S3 or locally.

        Args:
            model_name: Name of the model
            checkpoint_name: Name of the checkpoint

        Returns:
            Checkpoint state dict if found, None otherwise
        """
        local_path = Path(f"./tmp/checkpoints/{model_name}/{checkpoint_name}.pt")

        # Try to download from S3 if not found locally
        if self.s3_available and not local_path.exists():
            try:
                s3_key = f"{self.checkpoint_prefix}/{model_name}/{checkpoint_name}.pt"
                local_path.parent.mkdir(parents=True, exist_ok=True)
                
                self.s3.download_file(
                    self.bucket,
                    s3_key,
                    str(local_path)
                )
            except ClientError:
                return None

        # Load checkpoint if it exists
        if local_path.exists():
            return torch.load(local_path)
        
        return None

# Example usage
if __name__ == "__main__":
    storage = AWSModelStorage(
        bucket="my-model-bucket",
        region="us-west-2"
    )
    
    # Example saving a model
    """
    storage.save_model(
        model,
        tokenizer,
        model_name="my-summarization-model",
        version="1.0.0",
        metadata={
            "description": "Fine-tuned BART model for summarization",
            "training_data": "news_articles_v1"
        }
    )
    """
