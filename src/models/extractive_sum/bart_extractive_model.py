from dataclasses import dataclass
from typing import List, Dict, Optional, Union
import torch
import torch.nn.functional as F
from transformers import BartTokenizer, BartModel
import numpy as np
from nltk.tokenize import sent_tokenize
import logging
import nltk
from pathlib import Path

@dataclass
class BartExtractiveConfig:
    """Configuration for BART-based extractive summarization."""
    model_name: str = "facebook/bart-large"
    max_length: int = 1024
    top_k: int = 3
    batch_size: int = 8
    use_gpu: bool = True
    temperature: float = 1.0
    diversity_penalty: float = 0.5
    cache_dir: Optional[Path] = None

class BartExtractiveSummarizer:
    """
    BART-based extractive summarizer that uses the encoder
    to generate sentence embeddings and select important sentences.
    """
    
    def __init__(
        self,
        config: BartExtractiveConfig,
        logger: Optional[logging.Logger] = None
    ):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() and config.use_gpu else "cpu")
        
        try:
            # Initialize BART model and tokenizer
            self.model = BartModel.from_pretrained(
                config.model_name,
                cache_dir=config.cache_dir,
                device_map="auto" if torch.cuda.device_count() > 1 else None,
                torch_dtype=torch.bfloat16
            ).to(self.device)
            
            self.tokenizer = BartTokenizer.from_pretrained(
                config.model_name,
                cache_dir=config.cache_dir
            )
            
            # Download NLTK data if needed
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt')
                
        except Exception as e:
            self.logger.error(f"Failed to initialize BART model: {str(e)}")
            raise

    def get_sentence_embeddings(
        self,
        sentences: List[str],
        batch_size: Optional[int] = None
    ) -> torch.Tensor:
        """Generate sentence embeddings using BART encoder."""
        batch_size = batch_size or self.config.batch_size
        all_embeddings = []
        
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            
            with torch.no_grad():
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_length,
                    return_tensors="pt"
                ).to(self.device)
                
                outputs = self.model.encoder(**inputs)
                # Use mean pooling over the sequence length
                embeddings = outputs.last_hidden_state.mean(dim=1)
                all_embeddings.append(embeddings)
                
        return torch.cat(all_embeddings, dim=0)

    def calculate_sentence_scores(
        self,
        embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate importance scores for sentences using embeddings.
        Uses cosine similarity and diversity penalty.
        """
        # Calculate cosine similarity matrix
        cos_sim = F.cosine_similarity(
            embeddings.unsqueeze(1),
            embeddings.unsqueeze(0),
            dim=-1
        )
        
        # Apply temperature scaling
        cos_sim = cos_sim / self.config.temperature
        
        # Calculate sentence scores (mean similarity with other sentences)
        scores = cos_sim.mean(dim=1)
        
        # Apply diversity penalty
        if self.config.diversity_penalty > 0:
            penalties = cos_sim.max(dim=1)[0] * self.config.diversity_penalty
            scores = scores - penalties
            
        return scores

    def select_top_sentences(
        self,
        sentences: List[str],
        scores: torch.Tensor
    ) -> List[str]:
        """Select top-k sentences while maintaining original order."""
        if len(sentences) <= self.config.top_k:
            return sentences
            
        # Get indices of top-k scores
        top_indices = torch.topk(scores, min(self.config.top_k, len(scores))).indices
        # Sort indices to maintain original sentence order
        top_indices = sorted(top_indices.cpu().numpy())
        
        return [sentences[i] for i in top_indices]

    def summarize(self, text: Union[str, List[str]]) -> Dict[str, List[str]]:
        """
        Generate extractive summary using BART embeddings.
        
        Args:
            text: Input text or list of sentences to summarize
            
        Returns:
            Dictionary containing summary and original sentences
        """
        try:
            # Handle input text
            if isinstance(text, str):
                sentences = sent_tokenize(text)
            else:
                sentences = text
                
            if not sentences:
                raise ValueError("Input text is empty")
            
            # Generate embeddings
            embeddings = self.get_sentence_embeddings(sentences)
            
            # Calculate sentence scores
            scores = self.calculate_sentence_scores(embeddings)
            
            # Select top sentences
            summary = self.select_top_sentences(sentences, scores)
            
            return {
                "summary": summary,
                "original_sentences": sentences,
                "scores": scores.cpu().numpy().tolist()
            }
            
        except Exception as e:
            self.logger.error(f"Summarization failed: {str(e)}")
            raise

    def __call__(self, text: Union[str, List[str]]) -> Dict[str, List[str]]:
        """Callable interface for the summarizer."""
        return self.summarize(text)

    @torch.no_grad()
    def batch_summarize(
        self,
        texts: List[str],
        batch_size: Optional[int] = None
    ) -> List[Dict[str, List[str]]]:
        """
        Batch process multiple texts.
        
        Args:
            texts: List of input texts
            batch_size: Optional batch size override
            
        Returns:
            List of summary dictionaries
        """
        batch_size = batch_size or self.config.batch_size
        summaries = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_summaries = [self.summarize(text) for text in batch]
            summaries.extend(batch_summaries)
            
            # Clear GPU cache periodically
            if (i + 1) % (batch_size * 5) == 0:
                torch.cuda.empty_cache()
                
        return summaries
