# ZKML-Guard Docker Container
# Includes all dependencies for training, evaluation, and proof generation

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    curl \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /workspace

# Install Python dependencies
COPY requirements.txt /workspace/
RUN pip3 install --no-cache-dir -r requirements.txt

# Install EZKL
RUN curl -L https://github.com/zkonduit/ezkl/releases/download/v12.0.0/ezkl-linux-amd64 -o /usr/local/bin/ezkl \
    && chmod +x /usr/local/bin/ezkl

# Copy source code
COPY src/ /workspace/src/
COPY scripts/ /workspace/scripts/
COPY data/ /workspace/data/
COPY models/ /workspace/models/
COPY tests/ /workspace/tests/

# Set Python path
ENV PYTHONPATH=/workspace:$PYTHONPATH

# Default command
CMD ["/bin/bash"]

# Labels
LABEL maintainer="zkml-guard@example.com"
LABEL description="ZKML-Guard: Verifiable Inference for Blind Signing Prevention"
LABEL version="1.0.0"

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python3 -c "import torch; import numpy; print('OK')" || exit 1

# Example commands:
# Build: docker build -t zkml-guard:latest .
# Run: docker run --gpus all -it zkml-guard:latest
# Train: docker run --gpus all -v $(pwd)/data:/workspace/data zkml-guard:latest python3 scripts/train_model.py --data /workspace/data/training
# Evaluate: docker run --gpus all zkml-guard:latest python3 scripts/evaluate.py --model models/pytorch/zkml_guard.pth --test-data data/evaluation_dataset
