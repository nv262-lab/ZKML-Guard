"""
ZKML-Guard Transaction Risk Classification Model

This module implements the 4-layer feedforward neural network
optimized for Zero-Knowledge Machine Learning compatibility.

Architecture:
- Input: Feature vector (variable size based on extraction)
- Layer 1: 128 neurons + ReLU
- Layer 2: 256 neurons + ReLU  
- Layer 3: 128 neurons + ReLU
- Layer 4: 64 neurons + ReLU
- Output: 5-class softmax (Safe, Low Risk, Medium Risk, High Risk, Critical)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict
import numpy as np


class ZKMLGuardModel(nn.Module):
    """
    Transaction risk classification model optimized for ZKML.
    
    Design choices:
    - Uses ReLU instead of GELU/Swish for circuit efficiency
    - No batch normalization (complicates ZK circuits)
    - No dropout in inference mode
    - 8-bit quantization compatible
    """
    
    def __init__(self, input_dim: int = 42, num_classes: int = 5, dropout_rate: float = 0.3):
        """
        Initialize the ZKML-Guard model.
        
        Args:
            input_dim: Number of input features (default 42 based on feature engineering)
            num_classes: Number of risk classes (default 5)
            dropout_rate: Dropout rate for training (default 0.3)
        """
        super(ZKMLGuardModel, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Layer 1: Input -> 128
        self.fc1 = nn.Linear(input_dim, 128)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Layer 2: 128 -> 256
        self.fc2 = nn.Linear(128, 256)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Layer 3: 256 -> 128
        self.fc3 = nn.Linear(256, 128)
        self.dropout3 = nn.Dropout(dropout_rate)
        
        # Layer 4: 128 -> 64
        self.fc4 = nn.Linear(128, 64)
        self.dropout4 = nn.Dropout(dropout_rate)
        
        # Output layer: 64 -> num_classes
        self.fc_out = nn.Linear(64, num_classes)
        
        # Initialize weights using Xavier initialization
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights using Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Tuple of (logits, probabilities):
                - logits: Raw output scores of shape (batch_size, num_classes)
                - probabilities: Softmax probabilities of shape (batch_size, num_classes)
        """
        # Layer 1
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        # Layer 2
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        # Layer 3
        x = self.fc3(x)
        x = F.relu(x)
        x = self.dropout3(x)
        
        # Layer 4
        x = self.fc4(x)
        x = F.relu(x)
        x = self.dropout4(x)
        
        # Output layer
        logits = self.fc_out(x)
        probabilities = F.softmax(logits, dim=1)
        
        return logits, probabilities
    
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Make predictions with confidence scores.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Tuple of (predicted_class, confidence, probabilities):
                - predicted_class: Predicted risk class indices
                - confidence: Confidence scores for predictions
                - probabilities: Full probability distribution
        """
        self.eval()
        with torch.no_grad():
            _, probabilities = self.forward(x)
            confidence, predicted_class = torch.max(probabilities, dim=1)
            
        return predicted_class, confidence, probabilities
    
    def get_risk_label(self, class_idx: int) -> str:
        """
        Convert class index to risk label.
        
        Args:
            class_idx: Class index (0-4)
            
        Returns:
            Risk label string
        """
        labels = ["Safe", "Low Risk", "Medium Risk", "High Risk", "Critical"]
        return labels[class_idx]
    
    def quantize_for_zkml(self, bits: int = 8) -> 'ZKMLGuardModel':
        """
        Quantize model for ZKML compatibility.
        
        Args:
            bits: Number of bits for quantization (default 8)
            
        Returns:
            Quantized model
        """
        # Apply dynamic quantization
        quantized_model = torch.quantization.quantize_dynamic(
            self,
            {nn.Linear},
            dtype=torch.qint8
        )
        return quantized_model
    
    def export_to_onnx(self, filepath: str, example_input: torch.Tensor = None):
        """
        Export model to ONNX format for EZKL.
        
        Args:
            filepath: Output filepath for ONNX model
            example_input: Example input tensor (if None, uses dummy input)
        """
        self.eval()
        
        if example_input is None:
            example_input = torch.randn(1, self.input_dim)
        
        torch.onnx.export(
            self,
            example_input,
            filepath,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['logits', 'probabilities'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'logits': {0: 'batch_size'},
                'probabilities': {0: 'batch_size'}
            }
        )
        print(f"Model exported to {filepath}")
    
    @classmethod
    def load_pretrained(cls, filepath: str, input_dim: int = 42) -> 'ZKMLGuardModel':
        """
        Load pre-trained model weights.
        
        Args:
            filepath: Path to saved model weights
            input_dim: Input dimension (must match saved model)
            
        Returns:
            Loaded model
        """
        model = cls(input_dim=input_dim)
        model.load_state_dict(torch.load(filepath))
        model.eval()
        return model
    
    def save_model(self, filepath: str):
        """
        Save model weights.
        
        Args:
            filepath: Output filepath for model weights
        """
        torch.save(self.state_dict(), filepath)
        print(f"Model saved to {filepath}")


class ZKMLGuardEnsemble(nn.Module):
    """
    Ensemble of ZKML-Guard models for improved robustness.
    
    Note: This is primarily for training/validation. 
    For ZKML deployment, use single model for proof efficiency.
    """
    
    def __init__(self, num_models: int = 3, input_dim: int = 42, num_classes: int = 5):
        """
        Initialize ensemble.
        
        Args:
            num_models: Number of models in ensemble
            input_dim: Input feature dimension
            num_classes: Number of output classes
        """
        super(ZKMLGuardEnsemble, self).__init__()
        
        self.models = nn.ModuleList([
            ZKMLGuardModel(input_dim=input_dim, num_classes=num_classes)
            for _ in range(num_models)
        ])
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through ensemble (averages predictions).
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (averaged_logits, averaged_probabilities)
        """
        all_logits = []
        all_probs = []
        
        for model in self.models:
            logits, probs = model(x)
            all_logits.append(logits)
            all_probs.append(probs)
        
        # Average predictions
        avg_logits = torch.stack(all_logits).mean(dim=0)
        avg_probs = torch.stack(all_probs).mean(dim=0)
        
        return avg_logits, avg_probs
    
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Make ensemble predictions."""
        self.eval()
        with torch.no_grad():
            _, probabilities = self.forward(x)
            confidence, predicted_class = torch.max(probabilities, dim=1)
            
        return predicted_class, confidence, probabilities


def create_model(config: Dict = None) -> ZKMLGuardModel:
    """
    Factory function to create ZKML-Guard model.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Initialized model
    """
    if config is None:
        config = {
            'input_dim': 42,
            'num_classes': 5,
            'dropout_rate': 0.3
        }
    
    return ZKMLGuardModel(**config)


if __name__ == "__main__":
    # Example usage
    model = ZKMLGuardModel(input_dim=42, num_classes=5)
    
    # Create dummy input
    dummy_input = torch.randn(1, 42)
    
    # Forward pass
    logits, probs = model(dummy_input)
    print(f"Logits shape: {logits.shape}")
    print(f"Probabilities shape: {probs.shape}")
    print(f"Probabilities: {probs}")
    
    # Make prediction
    pred_class, confidence, _ = model.predict(dummy_input)
    print(f"\nPredicted class: {pred_class.item()}")
    print(f"Risk level: {model.get_risk_label(pred_class.item())}")
    print(f"Confidence: {confidence.item():.4f}")
    
    # Export to ONNX
    model.export_to_onnx("zkml_guard.onnx", dummy_input)
