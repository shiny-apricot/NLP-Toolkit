"""Module for loading pre-trained summarization models.

This module provides functionality to load pre-trained models from Hugging Face
for text summarization tasks.
"""

from all_dataclass import LoadPretrainedModelResult

from transformers.models.auto.modeling_auto import AutoModelForSeq2SeqLM
from transformers.models.auto.configuration_auto import AutoConfig

from typing import Any

def load_pretrained_model(
    model_name: str,
    logger: Any
) -> LoadPretrainedModelResult:
    """Load a pre-trained summarization model.

    Args:
        model_name: The name of the pre-trained model to load.
        logger: Logger instance for logging.

    Returns:
        LoadPretrainedModelResult: Dataclass containing the loaded model and configuration.
    """
    logger.info(f"Loading pre-trained model: {model_name}")
    
    # Load the model configuration
    model_config = AutoConfig.from_pretrained(model_name)
    
    # Load the pre-trained model
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    logger.info(f"Successfully loaded pre-trained model: {model_name}")
    
    return LoadPretrainedModelResult(
        model=model,
        model_config=model_config
    )
