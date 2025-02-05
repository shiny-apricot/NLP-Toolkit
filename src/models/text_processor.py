"""Text processing utilities for summarization."""

from typing import List
import torch
from transformers import PreTrainedTokenizer

class TextProcessor:
    """Handles text preprocessing and chunking."""
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        enable_chunking: bool = True
    ) -> None:
        self.tokenizer = tokenizer
        self.enable_chunking = enable_chunking

    def preprocess_text(self, text: str) -> str:
        """Clean and normalize input text."""
        text = ' '.join(text.split())
        text = text.replace('\u200b', '')
        return text

    def chunk_text(self, text: str, max_length: int) -> List[str]:
        """Split text into overlapping chunks."""
        if not self.enable_chunking:
            return [text]
            
        tokens = self.tokenizer.encode(text)
        if len(tokens) <= max_length:
            return [text]

        chunks = []
        stride = max_length // 2
        for i in range(0, len(tokens), stride):
            chunk_tokens = tokens[i:i + max_length]
            chunks.append(self.tokenizer.decode(chunk_tokens))
        
        return chunks

    def tokenize_batch(
        self,
        texts: List[str],
        max_length: int,
        device: torch.device
    ) -> dict:
        """Tokenize a batch of texts."""
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(device)
