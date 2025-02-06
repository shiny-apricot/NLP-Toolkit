import logging
import json
import sys
from typing import Optional, Dict, Any, Union
from datetime import datetime
from pathlib import Path
from logging import LogRecord
import watchtower
import boto3
from boto3.session import Session
import json
from dataclasses import asdict


def _create_log_record_factory():
    """Create a LogRecord factory that supports extra fields."""
    old_factory = logging.getLogRecordFactory()
    
    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        record.extra_fields = {}
        return record
    
    return record_factory

# Set custom LogRecord factory
logging.setLogRecordFactory(_create_log_record_factory())

class ProjectLogger(logging.Logger):
    """Structured logger with optional CloudWatch integration."""
    
    def __init__(
        self,
        name: str,
        *,
        level: str = "INFO",
        log_file: Optional[Path] = None,
        cloudwatch_group: Optional[str] = None,
        cloudwatch_stream: Optional[str] = None,
        aws_region: Optional[str] = None
    ) -> None:
        super().__init__(name, level)
        
        if not name:
            raise ValueError("Logger name cannot be empty")
            
        if level not in logging._nameToLevel:
            raise ValueError(f"Invalid logging level: {level}")

        self.handlers = []

        # Add console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(self._create_formatter())
        self.addHandler(console_handler)

        # Add file handler if specified
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            try:
                file_handler = logging.FileHandler(log_file)
                file_handler.setFormatter(self._create_formatter())
                self.addHandler(file_handler)
            except Exception as e:
                self.error(f"Failed to setup file logging: {str(e)}")

        # Add CloudWatch handler if specified
        if cloudwatch_group:
            if not aws_region:
                raise ValueError("aws_region is required when using CloudWatch logging")
            
            try:
                session: Session = boto3.Session(region_name=aws_region)
                cloudwatch_handler = watchtower.CloudWatchLogHandler(
                    log_group=cloudwatch_group,
                    stream_name=cloudwatch_stream or datetime.now().strftime("%Y-%m-%d"),
                    boto3_session=session,
                    send_interval=60,  # Send logs every 60 seconds
                    create_log_group=True
                )
                cloudwatch_handler.setFormatter(self._create_formatter())
                self.addHandler(cloudwatch_handler)
            except Exception as e:
                self.error(f"Failed to setup CloudWatch logging: {str(e)}")

    def _create_formatter(self) -> logging.Formatter:
        """Create JSON formatter for structured logging."""
        class JsonFormatter(logging.Formatter):
            def format(self, record: LogRecord) -> str:
                # Create base log entry
                output = {
                    "timestamp": datetime.fromtimestamp(record.created).isoformat(),
                    "level": record.levelname,
                    "logger": record.name,
                    "message": record.getMessage(),
                    "location": f"{record.filename}:{record.lineno}"
                }
                
                # Safely get extra fields
                extra_fields = getattr(record, "extra_fields", {})
                if extra_fields:
                    output.update(extra_fields)
                
                return json.dumps(output, default=str)
        
        return JsonFormatter()

    def _log_with_extra(self, level: int, msg: object, *args: Any, **kwargs: Any) -> None:
        """Internal logging method with extra context."""
        if kwargs:
            # Create extra dict with our custom fields
            extra = {"extra_fields": kwargs}
            # Log with extra fields
            super().log(level, msg, *args, extra=extra)
        else:
            super().log(level, msg, *args)

    def info(self, msg: object, *args: Any, **kwargs: Any) -> None:
        """Log info level message with optional context."""
        self._log_with_extra(logging.INFO, msg, *args, **kwargs)

    def error(self, msg: object, *args: Any, **kwargs: Any) -> None:
        """Log error level message with optional context."""
        self._log_with_extra(logging.ERROR, msg, *args, **kwargs)

    def warning(self, msg: object, *args: Any, **kwargs: Any) -> None:
        """Log warning level message with optional context."""
        self._log_with_extra(logging.WARNING, msg, *args, **kwargs)

    def debug(self, msg: object, *args: Any, **kwargs: Any) -> None:
        """Log debug level message with optional context."""
        self._log_with_extra(logging.DEBUG, msg, *args, **kwargs)
    
    def save_results(self, results: Any, output_file: Path) -> None:
        """Save pipeline results to a JSON file.

        Args:
            results: Pipeline results to save
            output_file: Path to output JSON file

        """

        # Convert dataclass instances to dictionaries

        results_dict = asdict(results)

        with open(output_file, 'w') as f:
            json.dump(results_dict, f, indent=2)       

        self.info(f"Results saved to {output_file}")

def get_logger(
    name: str,
    *,
    level: str = "INFO",
    log_file: Optional[Path] = None,
    cloudwatch_group: Optional[str] = None,
    cloudwatch_stream: Optional[str] = None,
    aws_region: Optional[str] = None
) -> ProjectLogger:
    """
    Get a configured logger instance.

    Args:
        name: Logger name
        level: Logging level
        log_file: Optional file path for logging
        cloudwatch_group: Optional CloudWatch log group
        cloudwatch_stream: Optional CloudWatch stream name
        aws_region: AWS region for CloudWatch

    Returns:
        Configured ProjectLogger instance
        
    Raises:
        ValueError: If invalid parameters are provided
    """
    return ProjectLogger(
        name,
        level=level,
        log_file=log_file,
        cloudwatch_group=cloudwatch_group,
        cloudwatch_stream=cloudwatch_stream,
        aws_region=aws_region
    )

# Example usage:
if __name__ == "__main__":
    logger = get_logger(
        "summarization",
        level="INFO",
        log_file=Path("logs/summarization.log"),
        cloudwatch_group="/aws/summarization",
        aws_region="us-west-2"
    )
    
    logger.info("Processing batch", batch_id=123, memory_usage=1024)



