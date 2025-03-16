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
       --config configs/instances/r5.yaml \
       --save-model models/bart-cpu-optimized \
       --samples 1000
   ```

### AWS SageMaker Deployment

1. **Set up AWS credentials**
   - Configure your AWS CLI with appropriate credentials or use IAM roles
   - Ensure your account has permissions for SageMaker, S3, and CloudWatch

2. **Prepare your SageMaker notebook instance**
   - Launch a SageMaker notebook instance (ml.t3.medium is sufficient for setup)
   - Clone this repository into the notebook instance

3. **Deploy a model endpoint**
   ```python
   from src.aws.sagemaker_deployment import deploy_summarization_model
   
   endpoint_name = deploy_summarization_model(
       model_path="models/bart-cpu-optimized",
       instance_type="ml.g4dn.xlarge",
       region_name="us-west-2"
   )
   ```

4. **Invoke the endpoint**
   ```python
   from src.aws.sagemaker_client import summarize_text
   
   summary = summarize_text(
       text="Your long text to summarize goes here...",
       endpoint_name=endpoint_name,
       max_length=150,
       min_length=50
   )
   print(summary)
   ```

## 🔧 Configuration

The project uses YAML configuration files located in the `configs/` directory:

- `configs/models/` - Model-specific configurations (BART, T5, TextRank)
- `configs/instances/` - AWS instance type configurations
- `configs/datasets/` - Dataset configurations for fine-tuning

Example configuration for instance types:
```yaml
# configs/instances/r5.yaml
instance_type: "ml.r5.xlarge"
memory_optimization: true
batch_size: 8
quantization: "int8"
```

## 📊 Evaluation

Run evaluation on your summarization results:

```bash
python -m src.evaluation.run_metrics \
    --predictions path/to/predictions.json \
    --references path/to/references.json \
    --metrics rouge1,rouge2,rougeL,bertscore
```

## 📁 Project Structure

```
summarization-project/
├── src/
│   ├── data_processing/       # Dataset handling
│   ├── models/                # BART and TextRank implementations
│   ├── evaluation/            # Evaluation metrics
│   ├── aws/                   # AWS integration
│   └── utils/                 # Shared utilities
├── tests/                     # Test files
└── configs/                   # Configuration files
```

## 📝 Project Roadmap

### 1. Setup & Environment
- [ ] AWS Configuration & Environment Setup
- [ ] GitHub Copilot Installation and Configuration
- [ ] Project Structure Implementation
- [ ] Development Environment Setup (venv, dependencies)

### 2. Basic Implementation
1. **Abstractive Summarization with Pre-trained Models**
    - [ ] Implement T5/BART base models
    - [ ] Basic prompt engineering
    - [ ] Simple pipeline creation

2. **Extractive Summarization**
    - [ ] Implement basic extractive algorithms
    - [ ] Sentence ranking implementation
    - [ ] Text preprocessing utilities

3. **Evaluation Framework**
    - [ ] Basic ROUGE metric implementation
    - [ ] Testing infrastructure
    - [ ] Basic logging and monitoring

### 3. Advanced Features
1. **Model Fine-tuning**
    - [ ] Dataset preparation
    - [ ] Fine-tuning pipeline
    - [ ] Model evaluation

2. **Enhanced Metrics**
    - [ ] BERTScore implementation
    - [ ] BLEU score integration
    - [ ] Metrics comparison framework

## 📚 Resources

- [AWS SageMaker Documentation](https://docs.aws.amazon.com/sagemaker/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [ROUGE Metrics](https://pypi.org/project/rouge-score/)
- [TextRank Algorithm](https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📧 Contact

For any questions or feedback, please contact [your-email@example.com]

