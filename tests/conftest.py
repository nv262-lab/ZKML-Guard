"""Pytest configuration."""
import pytest
import numpy as np

@pytest.fixture
def mock_transaction():
    return {
        'to': '0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb',
        'from': '0x123456789abcdef',
        'value': '1000000000000000000',
        'data': '0x',
        'gas': '21000',
        'gasPrice': '50000000000',
        'timestamp': 1700000000,
        'nonce': 0
    }
