# Text Summarization MVP for AWS SageMaker

A comprehensive text summarization solution that implements both extractive (TextRank) and abstractive (BART) summarization approaches, optimized for AWS SageMaker deployment.

## 🌟 Project Overview

This project provides tools for automatic text summarization using state-of-the-art techniques:
- **Abstractive Summarization**: Uses pre-trained transformer models (BART/T5) to generate new summaries
- **Extractive Summarization**: Implements TextRank algorithm to extract key sentences from the original text
- **Evaluation Framework**: ROUGE metrics to quantitatively assess summary quality

## 📋 Features

- Pre-trained model inference with BART and T5
- TextRank-based extractive summarization
- Fine-tuning capabilities on custom datasets
- Comprehensive evaluation metrics (ROUGE, BERTScore)
- AWS SageMaker optimization and deployment tools
- Easy configuration system for different instance types

## 🚀 Quick Start

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Summarization-AWS-project.git
   cd Summarization-AWS-project
   ```

2. **Set up a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Run a basic summarization example**
   ```bash
   python -m src.pipelines.run_summary_pretrained \
       --config configs/test_config.yaml \
   ```
