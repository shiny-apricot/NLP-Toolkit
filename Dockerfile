# Start with Python base image for CPU
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=UTC

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Install AWS CLI v1 (more stable with these package versions)
RUN pip3 install --no-cache-dir awscli==1.27.137

# Install dependencies in order
COPY constraints.txt requirements.txt ./
RUN pip3 install -r constraints.txt && \
    pip3 install -r requirements.txt

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p \
    cache/models \
    cache/datasets \
    logs \
    outputs \
    data

# Set Python path
ENV PYTHONPATH=/app:$PYTHONPATH

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

ENTRYPOINT ["./scripts/docker-entrypoint.sh"]
CMD ["train"]