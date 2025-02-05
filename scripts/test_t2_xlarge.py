"""Test script for t2.xlarge setup verification"""

from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModel
from src.utils.project_logger import get_logger

def verify_setup():
    """Verify the setup is working correctly."""
    logger = get_logger("setup_verification")
    
    try:
        # Test CPU setup
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"Number of CPU threads: {torch.get_num_threads()}")
        
        # Test model loading
        model_name = "facebook/bart-base"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        
        # Test tokenization
        text = "This is a test sentence."
        tokens = tokenizer(text, return_tensors="pt")
        
        logger.info("Setup verification completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Setup verification failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = verify_setup()
    exit(0 if success else 1)
