"""Tests for ZKML-Guard model."""
import pytest
import torch
from src.model.zkml_guard_model import ZKMLGuardModel

def test_model_creation():
    model = ZKMLGuardModel(input_dim=42, num_classes=5)
    assert model is not None

def test_model_forward():
    model = ZKMLGuardModel(input_dim=42, num_classes=5)
    x = torch.randn(10, 42)
    logits, probs = model(x)
    assert logits.shape == (10, 5)
    assert probs.shape == (10, 5)
