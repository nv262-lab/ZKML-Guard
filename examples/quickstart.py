"""Quickstart example for ZKML-Guard."""
from src.model.zkml_guard_model import ZKMLGuardModel
from src.feature_extraction.transaction_features import TransactionFeatureExtractor
import torch

# Load model
model = ZKMLGuardModel.load_pretrained('models/pytorch/zkml_guard.pth')

# Extract features
extractor = TransactionFeatureExtractor()
tx = {'to': '0xtest', 'from': '0xfrom', 'value': '0', 'data': '0x', 
      'gas': '21000', 'gasPrice': '1000000000', 'timestamp': 1700000000, 'nonce': 0}
features = extractor.extract_features(tx)

# Predict
features_tensor = torch.FloatTensor(features).unsqueeze(0)
pred_class, confidence, _ = model.predict(features_tensor)

print(f"Risk: {model.get_risk_label(pred_class.item())}")
print(f"Confidence: {confidence.item():.2%}")
