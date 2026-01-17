# Deployment Guide

## Production Deployment

### 1. Infrastructure Requirements
- GPU: NVIDIA A100 or RTX 4090 recommended
- CPU: 32+ cores for CPU-only deployment
- RAM: 16GB minimum
- Storage: 10GB for models and data

### 2. Installation
```bash
docker pull zkmlguard/zkml-guard:latest
```

### 3. Configuration
Set environment variables for your MPC platform.

### 4. Monitoring
Use Prometheus metrics endpoint.
