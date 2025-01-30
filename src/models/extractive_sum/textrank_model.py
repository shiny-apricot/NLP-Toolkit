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

@dataclass
class TextRankConfig:
    """Configuration for TextRank summarization."""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    top_k: int = 3
    similarity_threshold: float = 0.3
    use_gpu: bool = True
    damping_factor: float = 0.85
    max_iterations: int = 100

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
            self.model = AutoModel.from_pretrained(config.model_name).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
            
            # Download NLTK data if needed
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt')
                
        except Exception as e:
            self.logger.error(f"Failed to initialize TextRank model: {str(e)}")
            raise

    def get_sentence_embeddings(self, sentences: List[str]) -> np.ndarray:
        """Generate embeddings for a list of sentences."""
        embeddings = []
        
        with torch.no_grad():
            for sentence in sentences:
                inputs = self.tokenizer(
                    sentence,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                ).to(self.device)
                
                outputs = self.model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
                embeddings.append(embedding)
                
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

    def __call__(self, text: str) -> Dict[str, List[str]]:
        """
        Callable interface for the summarizer.
        
        Returns:
            Dictionary containing original sentences and summary
        """
        summary = self.summarize(text)
        return {
            "summary": summary,
            "original_sentences": sent_tokenize(text)
        }
