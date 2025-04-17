"""
Extractive text summarization module.

This module provides functionality for extractive text summarization
using NLTK. It ranks sentences by importance and selects the most 
relevant ones to form a concise summary of the original text.
"""

import time
from dataclasses import dataclass
from typing import List, Dict
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.probability import FreqDist

# Download necessary NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

@dataclass
class ExtractiveSummaryResult:
    """Result of extractive summarization process."""
    summary: str
    original_length: int
    summary_length: int
    compression_ratio: float
    processing_time: float
    top_sentences_indices: List[int]

def preprocess_text(
    text: str
) -> List[str]:
    """Preprocess and tokenize text into sentences.
    
    Args:
        text: Input text to be preprocessed
    
    Returns:
        List of sentences from the text
    """
    # Clean text and split into sentences
    text = re.sub(r'\s+', ' ', text)
    sentences = sent_tokenize(text)
    return sentences

def calculate_word_frequencies(
    text: str
) -> Dict[str, int]:
    """Calculate word frequencies in the text.
    
    Args:
        text: Input text for frequency analysis
    
    Returns:
        Dictionary with words and their frequencies
    """
    # Tokenize words and remove stopwords
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower())
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
    
    # Calculate frequencies
    word_frequencies = FreqDist(filtered_words)
    return dict(word_frequencies)

def score_sentences(
    sentences: List[str],
    word_frequencies: Dict[str, int]
) -> Dict[int, float]:
    """Score sentences based on word frequencies.
    
    Args:
        sentences: List of sentences to score
        word_frequencies: Dictionary of word frequencies
    
    Returns:
        Dictionary with sentence indices and their scores
    """
    sentence_scores = {}
    
    for i, sentence in enumerate(sentences):
        score = 0
        words = word_tokenize(sentence.lower())
        for word in words:
            if word in word_frequencies:
                score += word_frequencies[word]
        
        # Normalize by sentence length to avoid bias toward longer sentences
        sentence_scores[i] = score / max(len(words), 1)
    
    return sentence_scores

def generate_extractive_summary(
    text: str,
    num_sentences: int = 3
) -> ExtractiveSummaryResult:
    """Generate an extractive summary from the input text.
    
    Args:
        text: Input text to summarize
        num_sentences: Number of sentences to include in summary
    
    Returns:
        ExtractiveSummaryResult dataclass with summary and metadata
    """
    start_time = time.time()
    
    # Tokenize text into sentences
    sentences = preprocess_text(text)
    
    # Calculate word frequencies
    word_frequencies = calculate_word_frequencies(text)
    
    # Score sentences
    sentence_scores = score_sentences(sentences, word_frequencies)
    
    # Select top sentences
    top_sentences_indices = sorted(
        sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_sentences]
    )
    
    # Create summary with sentences in original order
    summary_sentences = [sentences[i] for i in top_sentences_indices]
    summary = " ".join(summary_sentences)
    
    # Calculate metrics
    original_length = len(text)
    summary_length = len(summary)
    compression_ratio = summary_length / original_length if original_length > 0 else 0
    processing_time = time.time() - start_time
    
    return ExtractiveSummaryResult(
        summary=summary,
        original_length=original_length,
        summary_length=summary_length,
        compression_ratio=compression_ratio,
        processing_time=processing_time,
        top_sentences_indices=top_sentences_indices
    )
