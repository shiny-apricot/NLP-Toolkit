from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer
import torch
import logging
from nltk.tokenize import sent_tokenize
import nltk
import json
from pathlib import Path

class TextRankError(Exception):
    """Base exception for TextRank-related errors."""
    pass

class ModelInitializationError(TextRankError):
    """Raised when model initialization fails."""
    pass

class EmbeddingError(TextRankError):
    """Raised when embedding generation fails."""
    pass

@dataclass
class TextRankConfig:
    """Configuration for TextRank summarization."""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    top_k: int = 3
    similarity_threshold: float = 0.3
    use_gpu: bool = True
    damping_factor: float = 0.85
    max_iterations: int = 100
    batch_size: int = 32
    use_bfloat16: bool = True
    cache_dir: Optional[Path] = None

@dataclass
class SummarizationOutput:
    """Structured output for summarization results."""
    summary: List[str]
    original_sentences: List[str]
    sentence_scores: np.ndarray
    processing_time_ms: float

class TextRankSummarizer:
    """
    TextRank-based extractive summarizer using sentence embeddings.
    """
    
    def __init__(
        self,
        config: TextRankConfig,
        logger: Optional[logging.Logger] = None
    ):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() and config.use_gpu else "cpu")
        
        try:
            # Initialize model with AWS-optimized settings
            model_kwargs = {
                "torch_dtype": torch.bfloat16 if config.use_bfloat16 else torch.float32,
                "device_map": "auto" if config.use_gpu else None,
                "cache_dir": config.cache_dir
            }
            
            self.model = AutoModel.from_pretrained(config.model_name, **model_kwargs)
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.model_name,
                cache_dir=config.cache_dir
            )
            
            # Download NLTK data if needed
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)
                
        except Exception as e:
            raise ModelInitializationError(f"Failed to initialize model: {str(e)}")

    def get_sentence_embeddings(self, sentences: List[str]) -> np.ndarray:
        """Generate embeddings in batches for memory efficiency."""
        embeddings = []
        
        try:
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.config.use_bfloat16):
                for i in range(0, len(sentences), self.config.batch_size):
                    batch = sentences[i:i + self.config.batch_size]
                    inputs = self.tokenizer(
                        batch,
                        padding=True,
                        truncation=True,
                        return_tensors="pt"
                    ).to(self.device)
                    
                    outputs = self.model(**inputs)
                    batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                    embeddings.extend(batch_embeddings)
                    
                    # Log memory usage for AWS monitoring
                    if self.config.use_gpu:
                        self.logger.debug(json.dumps({
                            "event": "batch_processing",
                            "batch_size": len(batch),
                            "memory_used": torch.cuda.memory_allocated() / 1024**2
                        }))
                    
        except Exception as e:
            raise EmbeddingError(f"Failed to generate embeddings: {str(e)}")
            
        return np.array(embeddings)

    def build_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """Build similarity matrix from sentence embeddings."""
        similarity_matrix = cosine_similarity(embeddings)
        
        # Apply threshold
        similarity_matrix[similarity_matrix < self.config.similarity_threshold] = 0
        
        # Normalize matrix
        norm = similarity_matrix.sum(axis=1, keepdims=True)
        norm[norm == 0] = 1  # Avoid division by zero
        return similarity_matrix / norm

    def rank_sentences(self, similarity_matrix: np.ndarray) -> np.ndarray:
        """Apply TextRank algorithm to rank sentences."""
        graph = nx.from_numpy_array(similarity_matrix)
        scores = nx.pagerank(
            graph,
            alpha=self.config.damping_factor,
            max_iter=self.config.max_iterations
        )
        return np.array(list(scores.values()))

    def summarize(self, text: str) -> List[str]:
        """
        Generate extractive summary using TextRank.
        
        Args:
            text: Input text to summarize
            
        Returns:
            List of most important sentences
        """
        try:
            # Split text into sentences
            sentences = sent_tokenize(text)
            if len(sentences) <= self.config.top_k:
                return sentences
            
            # Generate embeddings
            embeddings = self.get_sentence_embeddings(sentences)
            
            # Build similarity matrix
            similarity_matrix = self.build_similarity_matrix(embeddings)
            
            # Rank sentences
            sentence_scores = self.rank_sentences(similarity_matrix)
            
            # Get top-k sentences while preserving order
            top_indices = np.argsort(sentence_scores)[-self.config.top_k:]
            top_indices = sorted(top_indices)  # Sort to maintain original order
            
            return [sentences[i] for i in top_indices]
            
        except Exception as e:
            self.logger.error(f"Summarization failed: {str(e)}")
            raise

    def __call__(self, text: str) -> SummarizationOutput:
        """Generate extractive summary with timing and structured output."""
        import time
        start_time = time.perf_counter()
        
        try:
            sentences = sent_tokenize(text)
            if len(sentences) <= self.config.top_k:
                return SummarizationOutput(
                    summary=sentences,
                    original_sentences=sentences,
                    sentence_scores=np.ones(len(sentences)),
                    processing_time_ms=0.0
                )
            
            embeddings = self.get_sentence_embeddings(sentences)
            similarity_matrix = self.build_similarity_matrix(embeddings)
            sentence_scores = self.rank_sentences(similarity_matrix)
            
            top_indices = np.argsort(sentence_scores)[-self.config.top_k:]
            top_indices = sorted(top_indices)
            summary = [sentences[i] for i in top_indices]
            
            processing_time = (time.perf_counter() - start_time) * 1000
            
            return SummarizationOutput(
                summary=summary,
                original_sentences=sentences,
                sentence_scores=sentence_scores,
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            self.logger.error(json.dumps({
                "event": "summarization_error",
                "error": str(e),
                "text_length": len(text)
            }))
            raise TextRankError(f"Summarization failed: {str(e)}")

    def __enter__(self):
        """Context manager for automatic resource cleanup."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up GPU memory when done."""
        if self.config.use_gpu:
            torch.cuda.empty_cache()
