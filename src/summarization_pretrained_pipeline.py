"""
Summarization Pipeline for Text Processing

This module provides tools to automatically create summaries from long texts using AI.
Think of it as an intelligent assistant that reads long articles and writes shorter versions.

Key Features:
- Processes both single texts and multiple texts at once
- Automatically manages computer memory and GPU resources
- Provides detailed metrics about the summary quality
- Handles errors gracefully with clear error messages

Example Usage:
    # Create a summarization tool
    pipeline = SummarizationPipeline("g4dn.xlarge")
    
    # Summarize a single article
    result = pipeline.summarize("Your long article text here...")
    print(result.summary)
    
    # Summarize multiple articles at once
    results = pipeline.summarize([
        "First article text...",
        "Second article text..."
    ])
"""

from dataclasses import dataclass
from typing import List, Optional, Union
from pathlib import Path
import time
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from src.utils.project_logger import get_logger, ProjectLogger
from src.utils.gpu_utils import GPUManager, GPUConfig
from src.utils.aws_utils import AWSModelStorage
from src.evaluation.metrics import SummarizationMetrics
from src.config_loader import ConfigLoader, InstanceConfig

@dataclass
class SummarizationOutput:
    summary: str
    original_length: int
    summary_length: int
    processing_time_ms: float
    metrics: Optional[dict] = None

class TokenLengthError(Exception):
    """Raised when input text exceeds model's maximum token length."""
    pass

class SummarizationPipeline:
    def __init__(
        self,
        instance_type: str,
        *,
        config_dir: str = "configs/aws_configs",
        device_map: Union[str, dict] = "auto",
        s3_bucket: Optional[str] = None,
        aws_region: Optional[str] = None
    ) -> None:
        """Initialize summarization pipeline."""
        # Initialize logger with explicit name and optional CloudWatch config
        self.logger = get_logger(
            "summarization-pipeline",
            level="INFO",
            cloudwatch_group="/aws/summarization" if s3_bucket else None,
            cloudwatch_stream=f"instance-{instance_type}",
            aws_region=aws_region
        )
        
        # Log initialization
        self.logger.info(
            "Initializing summarization pipeline",
            instance_type=instance_type,
            s3_bucket=s3_bucket,
            aws_region=aws_region
        )
        
        # Select device first
        self.device = self._select_device(device_map)
        
        # Initialize AWS storage if bucket provided
        self.aws_storage = None
        if s3_bucket and aws_region:
            self.aws_storage = AWSModelStorage(
                bucket=s3_bucket,
                region=aws_region,
                logger=self.logger
            )
        
        # Load instance configuration
        config_loader = ConfigLoader(config_dir)
        self.config = config_loader.load_instance_config(instance_type)
        
        # Initialize GPU manager with config values
        gpu_config = GPUConfig(
            min_memory_mb=self.config.memory_gb * 1024,
            max_memory_percent=0.9,
            strategy="data_parallel",
            prefer_bfloat16=self.config.fp16
        )
        self.gpu_manager = GPUManager(gpu_config, self.logger)
        
        self.max_length = self.config.max_length
        self.min_length = max(50, self.max_length // 4)  # sensible default
        
        # Initialize metrics calculator
        self.metrics = SummarizationMetrics(
            rouge_types=["rouge1", "rouge2", "rougeL"],
            use_stemming=True,
            device=self.device,
            logger=self.logger
        )
        
        # Initialize model and tokenizer
        self.model, self.tokenizer = self._initialize_model()

    def _select_device(self, device_map: Union[str, dict]) -> torch.device:
        """Select appropriate device based on configuration."""
        try:
            if isinstance(device_map, str) and device_map != "auto":
                return torch.device(device_map)
            
            device = self.gpu_manager.select_device()
            self.logger.info("Device selected", device=str(device))
            return device
            
        except Exception as e:
            self.logger.warning(
                "Device selection failed, falling back to CPU",
                error=str(e)
            )
            return torch.device("cpu")

    def _initialize_model(self) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Initialize the pre-trained model and tokenizer."""
        dtype = (
            torch.bfloat16 if self.config.dtype == "bfloat16"
            else torch.float16 if self.config.dtype == "float16"
            else torch.float32
        )
        
        try:
            # Try loading from AWS first if available
            if self.aws_storage:
                try:
                    model, tokenizer, _ = self.aws_storage.load_model(
                        model_name=self.config.model_name,
                        version="latest",
                        use_bfloat16=(self.config.dtype == "bfloat16")
                    )
                    model = model.to(self.device)
                    return model, tokenizer
                except Exception as e:
                    self.logger.warning(f"Failed to load from S3: {e}")
            
            # Fall back to loading from HuggingFace
            model = AutoModelForSeq2SeqLM.from_pretrained(
                self.config.model_name,
                torch_dtype=dtype,
                cache_dir=self.config.cache_dir
            ).to(self.device)
            
            tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                cache_dir=self.config.cache_dir
            )
            
            return model, tokenizer
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model initialization failed: {e}")

    def summarize(
        self,
        text: Union[str, List[str]],
        *,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        num_beams: int = 4,
        calculate_metrics: bool = True,
        reference_text: Optional[Union[str, List[str]]] = None
    ) -> Union[SummarizationOutput, List[SummarizationOutput]]:
        """
        Create a shorter version of your text(s).

        Think of this as asking an intelligent assistant to read and summarize
        one or more articles while keeping the most important points.

        Args:
            text: The article(s) you want to summarize. Can be one text or a list of texts.
            max_length: Maximum words in the summary (optional)
            min_length: Minimum words in the summary (optional)
            num_beams: How thorough the AI should be (higher = better but slower)
            calculate_metrics: Whether to measure summary quality
            reference_text: Original text(s) to compare summary quality against

        Returns:
            A summary result containing:
            - The generated summary
            - Length of original and summary
            - Processing time
            - Quality metrics (if requested)

        Example:
            # Summarize one article
            result = pipeline.summarize("Long article text...")
            print(result.summary)

            # Summarize multiple articles
            results = pipeline.summarize([
                "First article...",
                "Second article..."
            ])
            for r in results:
                print(r.summary)

        Common Issues:
            - If text is too long, you'll get a TokenLengthError
            - If processing multiple texts, make sure you have enough memory
            - For best results, clean your text of special characters first
        """
        is_batch = isinstance(text, list)
        texts = text if is_batch else [text]
        references = reference_text if isinstance(reference_text, list) else [reference_text] if reference_text else None
        
        max_length = max_length or self.max_length
        min_length = min_length or self.min_length

        start_time = time.perf_counter()
        results = []
        
        try:
            with torch.no_grad():
                inputs = self.tokenizer(
                    texts,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=self.tokenizer.model_max_length
                ).to(self.device)

                if inputs["input_ids"].shape[1] > self.tokenizer.model_max_length:
                    raise TokenLengthError(
                        f"Input length {inputs['input_ids'].shape[1]} exceeds "
                        f"maximum length {self.tokenizer.model_max_length}"
                    )
                
                with torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
                    outputs = self.model.generate(
                        **inputs,
                        max_length=max_length,
                        min_length=min_length,
                        num_beams=num_beams,
                        early_stopping=True
                    )

                summaries = self.tokenizer.batch_decode(
                    outputs,
                    skip_special_tokens=True
                )
                
                for idx, summary in enumerate(summaries):
                    metrics_dict = None
                    if calculate_metrics and references and references[idx]:
                        metrics_start = time.perf_counter()
                        metrics_dict = self.metrics.calculate_metrics(
                            summary, 
                            references[idx]
                        )
                        metrics_time = (time.perf_counter() - metrics_start) * 1000
                        self.logger.debug(f"Metrics calculation took {metrics_time:.2f}ms")
                    
                    results.append(SummarizationOutput(
                        summary=summary,
                        original_length=len(texts[idx]),
                        summary_length=len(summary),
                        processing_time_ms=(time.perf_counter() - start_time) * 1000,
                        metrics=metrics_dict
                    ))

        except Exception as e:
            self.logger.error(f"Summarization failed: {e}")
            raise

        return results[0] if not is_batch else results

    def save_model(
        self,
        path: Optional[Path] = None,
        version: str = "latest",
        metadata: Optional[dict] = None
    ) -> None:
        """
        Save model and tokenizer locally or to S3.
        
        Args:
            path: Optional local path to save model
            version: Version string for S3 storage
            metadata: Optional metadata about the model
        """
        if self.aws_storage:
            try:
                self.aws_storage.save_model(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    model_name=self.config.model_name,
                    version=version,
                    metadata=metadata,
                    local_path=path
                )
                return
            except Exception as e:
                self.logger.warning(f"Failed to save to S3, falling back to local save: {e}")
        
        if path:
            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
        else:
            self.logger.warning("No path provided and S3 storage failed - model not saved")

    def __enter__(self) -> 'SummarizationPipeline':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Clean up resources."""
        if hasattr(self, 'model'):
            self.model.cpu()  # Move model to CPU to free GPU memory
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        torch.cuda.empty_cache()  # Clear CUDA cache
