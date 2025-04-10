from transformers.models.auto.tokenization_auto import AutoTokenizer


from typing import Any


def load_tokenizer(
    model_name: str,
    logger: Any
) -> Any:
    """Load the tokenizer for the specified model.

    Args:
        model_name: The name of the model/tokenizer to load.
        logger: Logger instance for logging.

    Returns:
        Any: The loaded tokenizer.
    """
    logger.info(f"Loading tokenizer: {model_name}.")
    return AutoTokenizer.from_pretrained(model_name)