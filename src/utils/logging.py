import logging
import json
import sys
import time
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass
import watchtower
import boto3
from pathlib import Path
import threading
from datetime import datetime

@dataclass
class LogConfig:
    """Configuration for logging setup."""
    app_name: str
    log_level: str = "INFO"
    use_cloudwatch: bool = False
    aws_region: Optional[str] = None
    log_group: Optional[str] = None
    log_stream: Optional[str] = None
    local_log_path: Optional[Path] = None
    json_format: bool = True
    include_timestamp: bool = True
    include_caller: bool = True

class StructuredLogger:
    """
    Custom logger with structured output and AWS CloudWatch integration.
    
    Supports both local file logging and CloudWatch logging with JSON formatting.
    Thread-safe logging with context information.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, config: LogConfig):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance

    def __init__(self, config: LogConfig):
        if hasattr(self, 'initialized'):
            return
            
        self.config = config
        self.logger = logging.getLogger(config.app_name)
        self.logger.setLevel(getattr(logging, config.log_level.upper()))
        
        # Clear any existing handlers
        self.logger.handlers = []
        
        # Add handlers based on configuration
        self._setup_handlers()
        
        self.initialized = True

    def _setup_handlers(self):
        """Setup logging handlers based on configuration."""
        handlers = []
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(self._create_formatter())
        handlers.append(console_handler)
        
        # File handler
        if self.config.local_log_path:
            try:
                file_handler = logging.FileHandler(
                    self.config.local_log_path / f"{self.config.app_name}.log"
                )
                file_handler.setFormatter(self._create_formatter())
                handlers.append(file_handler)
            except Exception as e:
                print(f"Failed to setup file logging: {str(e)}")
        
        # CloudWatch handler
        if self.config.use_cloudwatch:
            try:
                cloudwatch_handler = self._setup_cloudwatch_handler()
                if (cloudwatch_handler):
                    handlers.append(cloudwatch_handler)
            except Exception as e:
                print(f"Failed to setup CloudWatch logging: {str(e)}")
        
        # Add all handlers to logger
        for handler in handlers:
            self.logger.addHandler(handler)

    def _create_formatter(self) -> logging.Formatter:
        """Create JSON formatter for structured logging."""
        def format_function(record):
            record_dict = {
                "message": record.getMessage(),
                "level": record.levelname,
                "logger": record.name
            }
            
            if self.config.include_timestamp:
                record_dict["timestamp"] = datetime.fromtimestamp(record.created).isoformat()
                
            if self.config.include_caller:
                record_dict.update({
                    "function": record.funcName,
                    "filename": record.filename,
                    "lineno": record.lineno
                })
                
            if hasattr(record, "extra_data"):
                record_dict.update(record.extra_data)
                
            return json.dumps(record_dict)
            
        return logging.Formatter(format_function)

    def _setup_cloudwatch_handler(self) -> Optional[watchtower.CloudWatchLogHandler]:
        """Setup AWS CloudWatch handler."""
        if not all([self.config.aws_region, self.config.log_group]):
            return None
            
        return watchtower.CloudWatchLogHandler(
            log_group=self.config.log_group,
            stream_name=self.config.log_stream or datetime.now().strftime("%Y-%m-%d"),
            boto3_session=boto3.Session(region_name=self.config.aws_region)
        )

    def _log(
        self,
        level: str,
        message: str,
        extra: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """Internal logging method with extra context."""
        if extra:
            extra_data = {**kwargs, **extra}
        else:
            extra_data = kwargs
            
        record = logging.LogRecord(
            name=self.logger.name,
            level=getattr(logging, level.upper()),
            pathname=sys._getframe(2).f_code.co_filename,
            lineno=sys._getframe(2).f_lineno,
            msg=message,
            args=(),
            exc_info=None
        )
        record.extra_data = extra_data
        
        self.logger.handle(record)

    def info(self, message: str, **kwargs):
        """Log info level message."""
        self._log("INFO", message, **kwargs)

    def error(self, message: str, **kwargs):
        """Log error level message."""
        self._log("ERROR", message, **kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning level message."""
        self._log("WARNING", message, **kwargs)

    def debug(self, message: str, **kwargs):
        """Log debug level message."""
        self._log("DEBUG", message, **kwargs)

    def critical(self, message: str, **kwargs):
        """Log critical level message."""
        self._log("CRITICAL", message, **kwargs)

def get_logger(
    app_name: str,
    config: Optional[LogConfig] = None
) -> StructuredLogger:
    """Get or create a logger instance."""
    if config is None:
        config = LogConfig(app_name=app_name)
    return StructuredLogger(config)

# Example usage
if __name__ == "__main__":
    # Local logging configuration
    local_config = LogConfig(
        app_name="summarization",
        log_level="INFO",
        local_log_path=Path("logs"),
        json_format=True
    )
    
    # CloudWatch logging configuration
    cloud_config = LogConfig(
        app_name="summarization",
        log_level="INFO",
        use_cloudwatch=True,
        aws_region="us-west-2",
        log_group="/aws/summarization",
        json_format=True
    )
    
    logger = get_logger("summarization", local_config)
    logger.info(
        "Processing batch",
        batch_id=123,
        batch_size=32,
        memory_usage=1024
    )
