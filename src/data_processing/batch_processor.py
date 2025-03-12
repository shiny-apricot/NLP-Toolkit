"""Batch processing utilities for dataset loading."""

from typing import Dict, List, Any, Optional, Iterator
import torch
from datasets import Dataset
from transformers import PreTrainedTokenizer
from datetime import datetime
from .dataset_types import ProcessedDataset
from .preprocessing import TokenizerError


def process_batch(
    batch: Dict[str, List[Any]],
    *,
    tokenizer: PreTrainedTokenizer,
    text_column: str,
    summary_column: str,
    max_length: Optional[int],
    truncation: bool,
    padding: bool
) -> List[Dict[str, Any]]:
    """Process a single batch of data."""
    try:
        texts = batch[text_column]
        summaries = batch[summary_column]
        
        # Filter out empty texts
        valid_indices = [i for i, t in enumerate(texts) if t and isinstance(t, str)]
        if not valid_indices:
            raise TokenizerError("No valid texts in batch")
            
        texts = [texts[i] for i in valid_indices]
        summaries = [summaries[i] for i in valid_indices]
        
        # Tokenize with error handling
        try:
            tokenized = tokenizer(
                texts,
                max_length=max_length,
                truncation=truncation,
                padding=padding,
                return_tensors="pt",
                return_token_type_ids=True,
                add_special_tokens=True
            )
            
            tokenized_summaries = tokenizer(
                summaries,
                max_length=max_length,
                truncation=truncation,
                padding=padding,
                return_tensors="pt",
                return_token_type_ids=True,
                add_special_tokens=True
            )
            
        except Exception as e:
            raise TokenizerError(f"Batch tokenization failed: {str(e)}") from e
        
        return [{
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "text": text,
            "summary": summary
        } for input_ids, attention_mask, labels, text, summary in zip(
            tokenized["input_ids"],
            tokenized["attention_mask"],
            tokenized_summaries["input_ids"],
            texts,
            summaries
        )]
        
    except Exception as e:
        raise TokenizerError(f"Batch processing failed: {str(e)}") from e


def process_dataset_in_batches(
    dataset: Dataset,
    *,
    tokenizer: PreTrainedTokenizer,
    text_column: str,
    summary_column: str,
    batch_size: int,
    max_length: Optional[int],
    truncation: bool,
    padding: bool
) -> ProcessedDataset:
    """Process dataset in memory-efficient batches."""
    start_time = datetime.now()
    
    processed_batches = []
    raw_texts = []
    raw_summaries = []
    
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i + batch_size]
        processed_batch = process_batch(
            batch,
            tokenizer=tokenizer,
            text_column=text_column,
            summary_column=summary_column,
            max_length=max_length,
            truncation=truncation,
            padding=padding
        )
        processed_batches.append(processed_batch)
        raw_texts.extend(batch[text_column])
        raw_summaries.extend(batch[summary_column])

    combined_dataset = Dataset.from_dict({
        k: [item[k] for batch in processed_batches for item in batch]
        for k in processed_batches[0][0].keys()
    })

    return ProcessedDataset(
        dataset=combined_dataset,
        num_samples=len(raw_texts),
        processing_time=(datetime.now() - start_time).total_seconds(),
        raw_texts=raw_texts,
        raw_summaries=raw_summaries,
        metadata={
            "max_length": max_length,
            "truncation": truncation,
            "padding": padding,
            "timestamp": datetime.utcnow().isoformat()
        }
    )
