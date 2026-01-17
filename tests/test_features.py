"""Tests for feature extraction."""
import pytest
import numpy as np
from src.feature_extraction.transaction_features import TransactionFeatureExtractor

def test_feature_extraction():
    extractor = TransactionFeatureExtractor()
    tx = {
        'to': '0x742d35Cc', 'from': '0x123456',
        'value': '1000000000000000000', 'data': '0x',
        'gas': '21000', 'gasPrice': '50000000000',
        'timestamp': 1700000000, 'nonce': 0
    }
    features = extractor.extract_features(tx)
    assert features.shape == (42,)
    assert not np.any(np.isnan(features))
