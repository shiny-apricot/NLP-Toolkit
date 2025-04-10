from all_dataclass import Metrics, PreprocessedDataset, TrainModelResult


import torch
from rouge_score import rouge_scorer


from typing import Any


def evaluate_model(
    train_model_result: TrainModelResult,
    dataset: PreprocessedDataset,
    tokenizer: Any,
    logger: Any
) -> Metrics:
    """Evaluate the trained model.

    Args:
        train_model_result: Result of the trained model.
        dataset: The dataset to use for evaluation.
        tokenizer: The tokenizer used for decoding.
        logger: Logger instance for logging.

    Returns:
        Metrics: Evaluation metrics.
    """
    logger.info("Evaluating the model.")
    model = train_model_result.model
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    predictions = []
    references = []

    logger.info("Generating predictions for evaluation dataset.")
    for i, example in enumerate(dataset.test_dataset):
        # Convert list to PyTorch tensor
        input_ids = torch.tensor(example["input_ids"]).unsqueeze(0)
        outputs = model.generate(input_ids, max_length=128, num_beams=5)
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Decode reference from label ids
        reference = tokenizer.decode(example["labels"], skip_special_tokens=True)

        predictions.append(prediction)
        references.append(reference)

        # Limit evaluation to a reasonable number to avoid long processing
        if i >= 100:
            break

    logger.info("Calculating ROUGE scores.")
    rouge1, rouge2, rougeL = 0.0, 0.0, 0.0
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        rouge1 += scores["rouge1"].fmeasure
        rouge2 += scores["rouge2"].fmeasure
        rougeL += scores["rougeL"].fmeasure

    num_samples = len(predictions)
    logger.info(f"Evaluation completed on {num_samples} samples.")
    return Metrics(
        rouge_1=rouge1 / num_samples,
        rouge_2=rouge2 / num_samples,
        rouge_L=rougeL / num_samples
    )