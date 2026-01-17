# API Reference

## Model API

### ZKMLGuardModel

Main classification model.

#### Methods
- `__init__(input_dim, num_classes, dropout_rate)`
- `forward(x)` - Forward pass
- `predict(x)` - Get predictions
- `export_to_onnx(filepath)` - Export to ONNX

## Feature Extraction API

### TransactionFeatureExtractor

Extracts 42 features from transactions.

#### Methods
- `extract_features(transaction)` - Returns numpy array (42,)

## ZKML API

### ProofGenerator
- `generate_proof(features)` - Generate ZK proof

### ProofVerifier  
- `verify(proof_path)` - Verify proof
