# Training Dataset

This directory contains the training dataset described in the paper.

## Overview

- **Size**: 15,847 transactions
- **Time Period**: January 2023 - December 2025
- **Purpose**: Training the ZKML-Guard classification model

## ⚠️ Important Note

Per the paper's Open Science section:

> "The full training dataset cannot be released due to data sharing agreements with custody providers that restrict redistribution."

However, we provide:
1. ✅ **Feature extraction pipeline** - Can process any transaction data
2. ✅ **Training scripts** - Complete training procedure
3. ✅ **Synthetic training data** - For testing the training pipeline
4. ✅ **Model weights** - Pre-trained model available

## Files

- `features.npy` - Feature matrix of shape (15847, 42) - **Synthetic**
- `labels.npy` - Labels array of shape (15847,) - **Synthetic**
- `README.md` - This file

## Synthetic Data Characteristics

The provided synthetic data mimics the statistical properties of the real training data:

### Class Distribution

| Risk Category | Count | Percentage |
|---------------|-------|------------|
| Safe (0)      | ~11,000 | ~69%     |
| Low Risk (1)  | ~2,500  | ~16%     |
| Medium Risk (2)| ~1,250  | ~8%      |
| High Risk (3) | ~800    | ~5%      |
| Critical (4)  | ~315    | ~2%      |

**Total**: 15,847 transactions (matches paper)

### Feature Distributions

The synthetic data includes:
- Realistic age distributions for addresses
- Proper correlations between risk level and key indicators
- Representative calldata patterns
- Appropriate temporal patterns
- Contract verification status distributions

## Training Configuration

From the paper:

### Hyperparameters
- **Optimizer**: Adam
- **Learning Rate**: 0.001
- **Batch Size**: 256
- **Dropout Rate**: 0.3
- **Early Stopping Patience**: 10 epochs

### Training Procedure
- **Cross-Validation**: 5-fold stratified
- **Sampling**: Weighted sampling for class imbalance
- **Validation Split**: 20% of training data
- **Temporal Separation**: Strict time gap between train/test

### Expected Results
- **Mean Accuracy**: 94.7% ± 1.2% (cross-validation)
- **Training Time**: 40-60 epochs typically
- **Convergence**: Early stopping around epoch 50

## Using Your Own Data

To train with your own transaction data:

### 1. Extract Features

```python
from src.feature_extraction import TransactionFeatureExtractor

extractor = TransactionFeatureExtractor()

# Process your transactions
features = []
labels = []

for tx, label in your_transactions:
    feature_vector = extractor.extract_features(tx)
    features.append(feature_vector)
    labels.append(label)

# Save in the same format
import numpy as np
np.save('data/training/features.npy', np.array(features))
np.save('data/training/labels.npy', np.array(labels))
```

### 2. Train Model

```bash
python scripts/train_model.py \
    --data data/training \
    --epochs 100 \
    --batch-size 256 \
    --learning-rate 0.001 \
    --cross-validate \
    --n-folds 5 \
    --output models/pytorch
```

## Data Requirements

For optimal performance, your training data should include:

### Minimum Requirements
- At least 10,000 transactions
- Representation across all 5 risk categories
- Temporal diversity (multiple time periods)
- Geographic diversity (different blockchains/networks)

### Recommended
- 15,000+ transactions (as in paper)
- Confirmed attack examples for Critical category
- Verified legitimate transactions for Safe category
- Manual security review of Medium/High Risk labels

### Labeling Criteria

| Category    | Criteria |
|-------------|----------|
| Safe        | Verified legitimate transaction, no red flags |
| Low Risk    | Minor unusual patterns, low-value, common operations |
| Medium Risk | Multiple minor flags OR one moderate flag |
| High Risk   | Significant red flags, flagged by heuristics |
| Critical    | Confirmed attacks OR multiple severe red flags |

## Feature Engineering

The training data should contain all 42 features:

### Critical Features for Detection
The following features are most important for attack detection:

1. **Delegatecall presence** (Feature 16)
   - Binary indicator
   - Most critical for Bybit-style attacks

2. **Contract verification** (Feature 37)
   - Unverified contracts = higher risk
   - Strong indicator when combined with new age

3. **Contract age** (Feature 38)
   - Newly deployed = suspicious
   - Critical when < 7 days

4. **Destination address age** (Feature 1)
   - Very new addresses = higher risk
   - Critical when < 10 days

5. **Pattern matching** (Feature 19)
   - Matches known safe patterns
   - Inverse indicator (low match = risk)

See `data/evaluation_dataset/README.md` for complete feature descriptions.

## Data Collection Sources

The real training data (not included) was collected from:

1. **Custody Provider Data**
   - Institutional transaction logs
   - Anonymized under data sharing agreements
   - Includes confirmed attacks and legitimate operations

2. **Public Incident Reports**
   - Forensic reports (Sygnia, NCC Group, etc.)
   - On-chain attack data (Etherscan, etc.)
   - CVE-documented vulnerabilities

3. **Security Research**
   - Academic papers on blockchain attacks
   - Red team exercises
   - Honeypot deployments

## Temporal Separation

**Critical**: Training and test data must be temporally separated:

- Training: January 2023 - October 2025
- Test: November 2025 - December 2025

This prevents:
- Forward-looking information leakage
- Overfitting to recent patterns
- Unrealistic performance estimates

## Class Imbalance Handling

The training script handles class imbalance through:

1. **Weighted Sampling**
   - Each class weighted inversely to frequency
   - Critical class gets 35x weight vs Safe class

2. **Stratified Splitting**
   - Each fold maintains class distribution
   - Prevents fold-specific bias

3. **Evaluation Metrics**
   - Per-class precision/recall
   - F1-score for balanced view
   - Focus on Critical class performance

## Testing the Training Pipeline

Even with synthetic data, you can test the full training pipeline:

```bash
# Train on synthetic data
python scripts/train_model.py \
    --data data/training \
    --epochs 10 \
    --batch-size 256 \
    --output models/test

# This verifies:
# ✓ Data loading works
# ✓ Model architecture is correct
# ✓ Training loop executes
# ✓ Checkpointing works
# ✓ Cross-validation runs
```

**Note**: Performance on synthetic data will be lower than reported in paper, as it lacks the complexity of real attack patterns.

## Pre-trained Model Available

Since the full training data cannot be shared, we provide:

✅ **Pre-trained model weights** (`models/pytorch/zkml_guard.pth`)
- Trained on the real 15,847 transaction dataset
- Achieves 94.7% accuracy on evaluation set
- Ready for inference and ZKML proof generation

## Questions?

- Feature extraction: See `src/feature_extraction/`
- Training procedure: See `scripts/train_model.py`
- Data sharing: Contact data-sharing@zkml-guard.org
- Security: Report to security@zkml-guard.org

## Citation

```bibtex
@article{zkmlguard2025,
  title={ZKML-Secured MPC Custody: Verifiable Inference for Blind Signing Prevention},
  author={[Authors]},
  year={2025}
}
```
