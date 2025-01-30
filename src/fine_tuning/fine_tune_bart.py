from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
import torch
from torch.utils.data import Dataset
from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
import logging
import json
from pathlib import Path
import boto3
from botocore.exceptions import ClientError

@dataclass
class BartFineTuningConfig:
    model_name: str = "facebook/bart-large-cnn"
    max_length: int = 1024
    min_length: int = 50
    batch_size: int = 8
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_steps: int = 500
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 4
    fp16: bool = True
    output_dir: str = "checkpoints"
    logging_dir: str = "logs"
    save_total_limit: int = 2

class BartFineTuner:
    def __init__(
        self,
        config: BartFineTuningConfig,
        dataset: Dataset,
        s3_bucket: Optional[str] = None
    ):
        self.config = config
        self.dataset = dataset
        self.s3_bucket = s3_bucket
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.setup_logging()
        self.initialize_model()

    def setup_logging(self):
        self.logger = logging.getLogger("bart_fine_tuning")
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            json.dumps({
                "timestamp": "%(asctime)s",
                "level": "%(levelname)s",
                "message": "%(message)s"
            })
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def initialize_model(self):
        try:
            self.model = BartForConditionalGeneration.from_pretrained(
                self.config.model_name,
                device_map="auto" if torch.cuda.device_count() > 1 else None,
                torch_dtype=torch.bfloat16 if self.config.fp16 else torch.float32
            )
            self.tokenizer = BartTokenizer.from_pretrained(self.config.model_name)
        except Exception as e:
            self.logger.error(f"Failed to initialize model: {str(e)}")
            raise

    def prepare_training_args(self) -> TrainingArguments:
        return TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            fp16=self.config.fp16,
            logging_dir=self.config.logging_dir,
            save_total_limit=self.config.save_total_limit,
            save_strategy="steps",
            evaluation_strategy="steps",
            load_best_model_at_end=True,
            report_to="tensorboard"
        )

    def fine_tune(self) -> Tuple[BartForConditionalGeneration, Dict[str, float]]:
        """
        Fine-tune the BART model on the provided dataset.
        
        Returns:
            Tuple of (fine-tuned model, training metrics)
            
        Raises:
            RuntimeError: If training fails
            ValueError: If dataset is invalid
        """
        try:
            training_args = self.prepare_training_args()
            data_collator = DataCollatorForSeq2Seq(
                tokenizer=self.tokenizer,
                model=self.model,
                padding=True
            )

            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=self.dataset,
                data_collator=data_collator,
                tokenizer=self.tokenizer
            )

            self.logger.info("Starting fine-tuning")
            train_result = trainer.train()
            metrics = train_result.metrics

            # Save model
            trainer.save_model()
            if self.s3_bucket:
                self._save_to_s3(self.config.output_dir)

            self.logger.info(f"Training completed. Metrics: {metrics}")
            return self.model, metrics

        except Exception as e:
            self.logger.error(f"Fine-tuning failed: {str(e)}")
            raise RuntimeError(f"Fine-tuning failed: {str(e)}")

    def _save_to_s3(self, local_path: str):
        """Save model files to S3"""
        try:
            s3 = boto3.client('s3')
            for file_path in Path(local_path).glob('**/*'):
                if file_path.is_file():
                    s3_key = str(file_path.relative_to(local_path))
                    s3.upload_file(str(file_path), self.s3_bucket, s3_key)
            self.logger.info(f"Model saved to s3://{self.s3_bucket}")
        except ClientError as e:
            self.logger.error(f"Failed to save to S3: {str(e)}")
            raise

if __name__ == "__main__":
    # Example usage
    config = BartFineTuningConfig()
    # Initialize with your dataset and optionally S3 bucket
    # fine_tuner = BartFineTuner(config, dataset, "my-s3-bucket")
    # model, metrics = fine_tuner.fine_tune()
