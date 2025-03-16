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

## 🏃‍♂️ Running Summarization Tasks

### Model Training on SageMaker

#### 1. Prepare for SageMaker Training

First, ensure you have AWS credentials properly configured:

```bash
# Configure AWS CLI credentials if not using IAM roles
aws configure
```

Prepare your dataset by uploading it to S3:

```bash
# Create bucket if it doesn't exist
aws s3 mb s3://summarization-models

# Upload dataset (example with CNN/DailyMail)
aws s3 cp datasets/cnn_dailymail s3://summarization-models/datasets/cnn_dailymail --recursive
```

#### 2. Launch Training Job

You can launch a training job using our provided launcher script:

```bash
python -m src.training.sagemaker_launcher \
  --job-name bart-training-job \
  --role-arn arn:aws:iam::123456789012:role/SageMakerExecutionRole \
  --instance-type ml.t3.medium \
  --instance-count 1 \
  --model-name facebook/bart-large-cnn
```

Additional options:
- `--pytorch-version`: Specify PyTorch version (default: 1.13.1)
- `--bucket`: S3 bucket for storing model artifacts (default: summarization-models)
- `--dataset-name`: Dataset to use for training (default: cnn_dailymail)

#### 3. Monitor Training Progress

Track your training job in the AWS SageMaker console or via AWS CLI:

```bash
# Get training job status
aws sagemaker describe-training-job --training-job-name bart-training-job

# Stream training logs
aws logs get-log-events \
  --log-group-name /aws/sagemaker/TrainingJobs \
  --log-stream-name bart-training-job
```

### Inference with Trained Models

#### 1. Deploy Model to an Endpoint

After training is complete, deploy your model to a SageMaker endpoint:

```python
from src.inference.deploy_model import deploy_summarization_endpoint

endpoint_name = deploy_summarization_endpoint(
    model_s3_path="s3://summarization-models/models/bart-training-job/output/model.tar.gz",
    instance_type="ml.g4dn.xlarge",
    endpoint_name="summarization-endpoint"
)
```

#### 2. Run Inference

Summarize texts using your deployed endpoint:

```python
from src.inference.summarize_text import summarize_with_endpoint

text = """Your long text to summarize goes here. It should be several 
sentences long to demonstrate the summarization capabilities effectively."""

summary = summarize_with_endpoint(
    text=text,
    endpoint_name="summarization-endpoint",
    max_length=150,
    min_length=50
)

print(summary)
```

#### 3. Local Inference (without SageMaker endpoint)

For development or testing, run inference locally:

```python
from src.inference.local_inference import summarize_text_locally

text = "Your long text to summarize goes here..."

summary = summarize_text_locally(
    text=text,
    model_path="models/bart-fine-tuned",
    max_length=150,
    min_length=50,
    device="cuda"  # or "cpu"
)

print(summary)
```

### Running Extractive Summarization with TextRank

TextRank provides a faster, computation-efficient summarization alternative:

```python
from src.models.extractive_sum.textrank import summarize_with_textrank

text = """Your long document to summarize with the extractive method. 
This approach will select the most important sentences rather than 
generating new content."""

summary = summarize_with_textrank(
    text=text,
    num_sentences=3,
    language="english"
)

print(summary)
```

### Advanced Configuration

Fine-tune your summarization tasks by editing configuration files:

```yaml
# Example: configs/training/bart_custom.yaml
model_name: facebook/bart-large-cnn
dataset_name: cnn_dailymail
num_epochs: 5
batch_size: 8
learning_rate: 3e-5
max_length: 1024
min_length: 50
```

Then use this configuration:

```bash
python -m src.training.train_with_config \
  --config configs/training/bart_custom.yaml \
  --output-dir models/bart-custom
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

