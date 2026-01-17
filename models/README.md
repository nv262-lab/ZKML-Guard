# Model Artifacts

## PyTorch Model
- `pytorch/zkml_guard.pth` - Trained weights (1.2 MB)
- Achieves 94.7% accuracy on evaluation set

## ONNX Model  
- `onnx/zkml_guard.onnx` - EZKL-compatible
- 8-bit quantized

## Verification Keys
- `verification_keys/vk.key` - For proof verification
- `verification_keys/srs.params` - Structured reference string

## Usage
```python
from src.model.zkml_guard_model import ZKMLGuardModel
model = ZKMLGuardModel.load_pretrained('models/pytorch/zkml_guard.pth')
```
