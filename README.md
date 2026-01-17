# ZKML-Guard: Verifiable Inference for Blind Signing Prevention in Digital Asset Management

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

This repository contains the implementation of **ZKML-Guard**, a framework that combines Zero-Knowledge Machine Learning (ZKML) with Multi-Party Computation (MPC) custody to prevent blind signing vulnerabilities in cryptocurrency transactions.

## 📄 Paper Reference

**ZKML-Secured MPC Custody: Verifiable Inference for Blind Signing Prevention in Digital Asset Management**

Our system achieved:
- **94.7% accuracy** in transaction risk classification
- **847ms median** proof generation time (GPU)
- **99.2% confidence** in detecting the Bybit attack vector
- **0.3% false positive rate**

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8 or higher
python --version

# Install dependencies
pip install -r requirements.txt

# Install EZKL (for zero-knowledge proof generation)
curl https://github.com/zkonduit/ezkl/releases/download/v12.0.0/ezkl-linux-amd64 -L -o ezkl
chmod +x ezkl
```

### Docker Setup (Recommended)

```bash
# Build Docker container
docker build -t zkml-guard .

# Run container
docker run -it --gpus all zkml-guard
```

## 📁 Repository Structure

```
zkml-guard/
├── data/
│   ├── public_samples/           # Public transaction samples
│   ├── evaluation_dataset/       # Complete evaluation dataset
│   └── feature_pipeline/         # Feature extraction code
├── models/
│   ├── pytorch/                  # PyTorch model weights
│   ├── onnx/                     # ONNX format models
│   └── verification_keys/        # ZKML verification keys
├── src/
│   ├── feature_extraction/       # Feature engineering modules
│   ├── model/                    # Neural network architecture
│   ├── zkml/                     # Zero-knowledge proof generation
│   └── integration/              # MPC custody integration
├── scripts/
│   ├── train_model.py           # Model training script
│   ├── generate_proof.py        # Proof generation script
│   └── evaluate.py              # Evaluation reproduction
├── tests/                        # Unit and integration tests
├── docker/                       # Docker configuration
└── docs/                        # Documentation

```

## 🔧 Usage

### 1. Feature Extraction

```python
from src.feature_extraction import TransactionFeatureExtractor

extractor = TransactionFeatureExtractor()
features = extractor.extract_features(transaction_data)
```

### 2. Model Inference

```python
from src.model import ZKMLGuardModel

model = ZKMLGuardModel.load_pretrained('models/pytorch/zkml_guard.pth')
risk_score, confidence = model.predict(features)
```

### 3. Generate Zero-Knowledge Proof

```python
from src.zkml import ProofGenerator

proof_gen = ProofGenerator(
    model_path='models/onnx/zkml_guard.onnx',
    srs_path='models/verification_keys/srs.params'
)

proof, public_output = proof_gen.generate_proof(features)
```

### 4. Verify Proof

```python
from src.zkml import ProofVerifier

verifier = ProofVerifier(vk_path='models/verification_keys/vk.key')
is_valid = verifier.verify(proof, public_output)
```

### 5. Integration with MPC Custody

```python
from src.integration import MPCCustodyIntegration

integration = MPCCustodyIntegration(
    proof_generator=proof_gen,
    verifier=verifier
)

# Intercept transaction proposal
result = integration.evaluate_transaction(raw_transaction)
if result.risk_level == "CRITICAL":
    print(f"High risk detected: {result.confidence}")
```

## 📊 Reproducing Results

### Training the Model

```bash
# Train from scratch
python scripts/train_model.py \
    --data data/training/ \
    --epochs 100 \
    --batch-size 256 \
    --learning-rate 0.001 \
    --output models/pytorch/

# Results will match paper: 94.7% ± 1.2% accuracy
```

### Running Evaluation

```bash
# Reproduce paper results
python scripts/evaluate.py \
    --model models/pytorch/zkml_guard.pth \
    --test-data data/evaluation_dataset/ \
    --output results/

# Expected output:
# Overall Accuracy: 94.7%
# Critical Precision: 97.3%
# Critical Recall: 91.8%
# False Positive Rate: 0.28%
```

### Proof Generation Benchmarks

```bash
# Benchmark proof generation
python scripts/benchmark_proof.py \
    --hardware GPU \
    --iterations 1000

# Expected results (NVIDIA A100):
# Median: 847ms
# P95: 1,124ms
# Memory: 4.2GB
```

### Bybit Attack Case Study

```bash
# Reproduce Bybit attack detection
python scripts/bybit_case_study.py \
    --attack-data data/public_samples/bybit_reconstruction.json

# Expected output:
# Displayed Transaction: Safe (98.1%)
# Actual Transaction: Critical (99.2%)
# Detection: SUCCESS
```

## 🔬 Model Architecture

The ZKML-Guard model is a 4-layer feedforward neural network optimized for ZKML compatibility:

```
Input Layer (Feature Vector)
    ↓
FC Layer 1: 128 neurons + ReLU
    ↓
FC Layer 2: 256 neurons + ReLU
    ↓
FC Layer 3: 128 neurons + ReLU
    ↓
FC Layer 4: 64 neurons + ReLU
    ↓
Softmax Output: 5 Risk Classes
    (Safe, Low Risk, Medium Risk, High Risk, Critical)
```

**Key Design Choices:**
- **No transformers/attention**: Optimized for ZK circuit efficiency
- **8-bit quantization**: Required for finite field arithmetic
- **ReLU activation**: Reduces circuit complexity vs. GELU/Swish

## 📦 Data Availability

### Public Dataset (Included)
- `data/public_samples/`: 500 labeled transactions from public incident reports
- `data/evaluation_dataset/`: Complete 3,169 transaction test set

### Feature Pipeline (Included)
The complete feature extraction pipeline can process any blockchain transaction:

```bash
# Extract features from on-chain data
python scripts/extract_features.py \
    --tx-hash 0xABCD... \
    --network ethereum \
    --output features.json
```

### Training Data (Restricted)
The full 15,847 transaction training dataset cannot be released due to data sharing agreements with custody providers. However:
- Feature extraction pipeline provided
- Training scripts provided
- Public samples can be used for testing

## 🔑 Model Artifacts

### PyTorch Weights
- **Location**: `models/pytorch/zkml_guard.pth`
- **Size**: 1.2 MB
- **SHA256**: `[to be computed]`

### ONNX Model
- **Location**: `models/onnx/zkml_guard.onnx`
- **Compatible with**: EZKL 12.0+
- **SHA256**: `[to be computed]`

### Verification Keys
- **SRS Parameters**: `models/verification_keys/srs.params` (45 seconds generation time)
- **Verification Key**: `models/verification_keys/vk.key`
- **Model Registry Hash**: `[to be registered on-chain]`

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific test suites
pytest tests/test_feature_extraction.py
pytest tests/test_model_inference.py
pytest tests/test_zkml_proof.py
pytest tests/test_integration.py

# Run with coverage
pytest --cov=src tests/
```

## 🐳 Docker Containers

Pre-built Docker containers are available with all dependencies:

```bash
# CPU-only version
docker pull zkmlguard/zkml-guard:cpu-v1.0

# GPU-enabled version (CUDA 11.8)
docker pull zkmlguard/zkml-guard:gpu-v1.0

# Run evaluation
docker run zkmlguard/zkml-guard:gpu-v1.0 python scripts/evaluate.py
```

## 📈 Performance Benchmarks

### Classification Performance (Test Set: 3,169 transactions)

| Risk Category | Precision | Recall | F1-Score | Support |
|---------------|-----------|--------|----------|---------|
| Safe          | 96.2%     | 97.8%  | 97.0%    | 2,241   |
| Low Risk      | 91.4%     | 89.2%  | 90.3%    | 412     |
| Medium Risk   | 88.7%     | 85.3%  | 87.0%    | 289     |
| High Risk     | 93.6%     | 90.1%  | 91.8%    | 156     |
| Critical      | 97.3%     | 91.8%  | 94.5%    | 71      |

**Overall Accuracy: 94.7%**

### Proof Generation Performance

| Configuration      | Median  | P95     | Memory |
|-------------------|---------|---------|--------|
| GPU (A100)        | 847 ms  | 1,124 ms| 4.2 GB |
| GPU (RTX 4090)    | 1,203 ms| 1,567 ms| 3.8 GB |
| CPU (Xeon 32-core)| 3,241 ms| 4,892 ms| 6.7 GB |
| CPU (16-core)     | 5,847 ms| 7,234 ms| 8.1 GB |

**Verification Time: 8ms (constant across all configurations)**

## 🔒 Security Considerations

### Responsible Disclosure
Our Bybit attack analysis uses only publicly available forensic reports:
- Sygnia Investigation Report (March 2025)
- NCC Group Technical Analysis (2025)
- On-chain transaction data

No unauthorized testing was performed on live systems.

### Model Privacy
The zero-knowledge property ensures:
- Model parameters remain confidential
- Transaction histories are not exposed
- Adversaries cannot query the model directly

### Dual-Use Awareness
While the model could theoretically be used to test attack evasion, the ZK property and well-known features prevent meaningful adversarial optimization.

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Areas of interest:
- Federated learning for privacy-preserving model updates
- On-chain verification integration
- Support for additional MPC custody platforms
- Performance optimizations for proof generation

## 📜 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## ⚠️ Limitations

- Model trained on data through December 2025
- Requires retraining for new attack patterns
- Training reproduction requires similar transaction datasets
- Performance benchmarks assume specific hardware configurations

## 🔄 Updates

- **v1.0.0** (2025-01): Initial release
- Model and verification keys published
- Evaluation dataset released
- Docker containers available

---

**Note**: This is research software. Use in production environments requires thorough security auditing and integration testing with your specific custody infrastructure.
