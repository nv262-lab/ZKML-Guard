# Evaluation Dataset

This directory contains the complete evaluation dataset used in the paper.

## Overview

- **Size**: 3,169 transactions
- **Time Period**: Temporally separated from training data
- **Purpose**: Reproducing Table 1 results from the paper

## Files

- `features.npy` - Feature matrix of shape (3169, 42)
- `labels.npy` - Labels array of shape (3169,)
- `README.md` - This file

## Class Distribution

Exact distribution from paper (Table 1):

| Risk Category | Count | Percentage |
|---------------|-------|------------|
| Safe (0)      | 2,241 | 70.7%      |
| Low Risk (1)  | 412   | 13.0%      |
| Medium Risk (2)| 289   | 9.1%       |
| High Risk (3) | 156   | 4.9%       |
| Critical (4)  | 71    | 2.2%       |

**Total**: 3,169 transactions

## Expected Results

Running evaluation on this dataset should produce:

### Overall Performance
- **Accuracy**: 94.7%
- **False Positive Rate**: 0.28%

### Per-Class Performance

| Category    | Precision | Recall | F1-Score |
|-------------|-----------|--------|----------|
| Safe        | 96.2%     | 97.8%  | 97.0%    |
| Low Risk    | 91.4%     | 89.2%  | 90.3%    |
| Medium Risk | 88.7%     | 85.3%  | 87.0%    |
| High Risk   | 93.6%     | 90.1%  | 91.8%    |
| Critical    | 97.3%     | 91.8%  | 94.5%    |

## Usage

### Load Dataset

```python
import numpy as np

# Load features and labels
features = np.load('features.npy')
labels = np.load('labels.npy')

print(f"Features shape: {features.shape}")  # (3169, 42)
print(f"Labels shape: {labels.shape}")      # (3169,)
```

### Run Evaluation

```bash
python scripts/evaluate.py \
    --model models/pytorch/zkml_guard.pth \
    --test-data data/evaluation_dataset \
    --output results/
```

## Feature Description

All 42 features per transaction:

### Reputation Features (12)
1. Destination address age (days)
2. Total transaction count (log-scaled)
3. Interaction diversity (log-scaled)
4. Known malicious flag (0/1)
5. Sanctioned address flag (0/1)
6. Contract deployment flag (0/1)
7. Source address age (days)
8. Source total transactions (log-scaled)
9. Source interaction diversity (log-scaled)
10. Previous interactions count (log-scaled)
11. Destination balance ETH (log-scaled)
12. Source balance ETH (log-scaled)

### Calldata Features (10)
13. Calldata length (log-scaled bytes)
14. Function selector (normalized)
15. Number of parameters
16. Delegatecall presence (0/1) **CRITICAL**
17. Nested call depth
18. Data complexity score
19. Unusual patterns flag (0/1)
20. Parameter type diversity
21. Data entropy
22. Known safe function flag (0/1)

### Value Distribution Features (8)
23. Transaction value ETH (log-scaled)
24. Value as % of sender balance
25. Value as % of receiver balance
26. Gas limit (log-scaled)
27. Gas price Gwei (log-scaled)
28. Total fee ETH (log-scaled)
29. Unusual gas limit flag (0/1)
30. Priority fee ratio

### Temporal Features (6)
31. Hour of day (0-23, normalized)
32. Day of week (0-6, normalized)
33. Time since last tx (minutes, log-scaled)
34. Transaction burst indicator (0/1)
35. Time deviation from pattern
36. Nonce gap indicator (0/1)

### Smart Contract Features (6)
37. Contract verified flag (0/1) **CRITICAL**
38. Contract age (days, log-scaled) **CRITICAL**
39. Proxy pattern indicator (0/1)
40. Recent contract modification (0/1)
41. Contract complexity score
42. Known vulnerability flag (0/1)

## Data Quality

- No missing values (all features present)
- All features normalized/scaled appropriately
- Temporal separation from training ensures no data leakage
- Ground truth labels verified through:
  - Confirmed attack reports
  - Manual security review
  - Post-incident forensic analysis

## Anonymization

All transaction data has been:
- Address hashed before inclusion
- No personally identifiable information (PII)
- Compliant with data sharing agreements
- Public attack vectors use only publicly documented information

## Reproducibility

This dataset enables exact reproduction of:
- Table 1 (Classification Performance)
- Figure 4 (Precision and Recall by Category)
- All confusion matrices
- False positive rate calculations

## Notes

⚠️ **Note**: This is the complete test set - do NOT use for training or hyperparameter tuning.

✅ **Validation**: This dataset was temporally separated from training to prevent data leakage.

📊 **Stratification**: Class distribution reflects real-world transaction patterns with heavy imbalance toward Safe transactions.

## Citation

If you use this dataset, please cite:

```bibtex
@article{zkmlguard2025,
  title={ZKML-Secured MPC Custody: Verifiable Inference for Blind Signing Prevention},
  author={[Authors]},
  year={2025}
}
```
