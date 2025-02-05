"""
Orchestrates the fine-tuning pipeline for BART summarization models.
Handles data loading, model training, evaluation and storage.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any
import torch

from src.models.bart_abstractive_model import BartAbstractiveSummarizer 
from src.data.loader import HuggingFaceLoader
from src.evaluation.metrics import SummarizationMetrics
from src.models.fine_tune_bart import BartFineTuner, BartFineTuningConfig
from src.utils.aws_utils import AWSModelStorage
from src.utils.gpu_utils import GPUManager, GPUConfig
from src.utils.project_logger import get_logger
from src.utils.save_model import save_model, ModelSaveError, load_metadata

@dataclass
class TrainingResult:
    """Results from model fine-tuning."""
    model_path: Path
    metrics: Dict[str, float] 
    training_duration: float
    device_used: str
    git_info: Optional[Dict[str, Any]] = None

class SummarizationFineTuningPipeline:
    """Pipeline for fine-tuning summarization models."""

    def __init__(
        self,
        *,  # Force named parameters
        dataset_name: str,
        model_name: str = "facebook/bart-large-cnn",
        s3_bucket: Optional[str] = None,
        aws_region: Optional[str] = "us-west-1",
        cache_dir: Optional[Path] = None,
        device_preference: str = "auto"
    ):
        """Initialize the fine-tuning pipeline."""
        self.logger = get_logger("fine_tuning")
        self.s3_bucket = s3_bucket
        self.aws_region = aws_region
        
        # Initialize GPU management
        gpu_config = GPUConfig(
            min_memory_mb=8 * 1024,  # 8GB minimum
            max_memory_percent=0.9,
            strategy="data_parallel",
            prefer_bfloat16=True
        )
        self.gpu_manager = GPUManager(gpu_config, self.logger)
        self.device = self.gpu_manager.select_device()

        # Initialize components
        self.dataset_loader = HuggingFaceLoader(
            dataset_name=dataset_name,
            text_column="text",
            summary_column="summary", 
            cache_dir=cache_dir,
            device=str(self.device)
        )

        self.metrics = SummarizationMetrics(
            use_stemming=True,
            batch_size=16,
            device=str(self.device)
        )

        # Initialize AWS storage if configured
        self.storage = None
        if s3_bucket:
            self.storage = AWSModelStorage(
                bucket=s3_bucket,
                region=aws_region,
                logger=self.logger
            )

        # Set training configuration
        self.training_config = BartFineTuningConfig(
            model_name=model_name,
            max_length=1024,
            min_length=50,
            batch_size=8,
            learning_rate=2e-5,
            num_epochs=3,
            warmup_steps=500,
            weight_decay=0.01,
            fp16=True
        )

    def run_pipeline(
        self,
        *,  # Force named parameters
        output_dir: Path,
        max_samples: Optional[int] = None,
        evaluate: bool = True,
        model_version: str,
        model_description: Optional[str] = None
    ) -> TrainingResult:
        """Run the full fine-tuning pipeline."""
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()

        try:
            # Load and prepare dataset
            dataset = self.dataset_loader.load(
                split="train",
                max_samples=max_samples,
                shuffle=True
            )

            # Initialize fine-tuning
            fine_tuner = BartFineTuner(
                config=self.training_config,
                dataset=dataset,
                s3_bucket=self.s3_bucket 
            )

            # Fine-tune the model
            model, metrics = fine_tuner.fine_tune()

            # Save the fine-tuned model with full metadata
            save_path = output_dir / "fine_tuned_model"
            try:
                save_path = save_model(
                    model=model,
                    tokenizer=fine_tuner.tokenizer,
                    save_path=save_path,
                    version=model_version,
                    description=model_description,
                    training_params=self.training_config.__dict__,
                    performance_metrics=metrics,
                    s3_bucket=self.s3_bucket,
                    aws_region=self.aws_region,
                    compress=True,
                    save_optimizer=True
                )
            except ModelSaveError as e:
                self.logger.error(f"Failed to save model: {str(e)}")
                raise

            end_time.record()
            torch.cuda.synchronize()
            duration = start_time.elapsed_time(end_time) / 1000  # Convert to seconds

            result = TrainingResult(
                model_path=save_path,
                metrics=metrics,
                training_duration=duration,
                device_used=str(self.device),
                git_info=load_metadata(save_path).git_info.__dict__ if load_metadata(save_path).git_info else None
            )

            self.logger.info(
                "Fine-tuning completed",
                extra={
                    "model_path": str(save_path),
                    "training_duration": duration,
                    "device": str(self.device),
                    "metrics": metrics,
                    "git_info": result.git_info
                }
            )

            return result

        except Exception as e:
            self.logger.error(f"Fine-tuning failed: {str(e)}")
            raise