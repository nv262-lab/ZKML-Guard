"""
ZKML-Guard: Verifiable Inference for Blind Signing Prevention

A framework combining Zero-Knowledge Machine Learning (ZKML) with 
Multi-Party Computation (MPC) custody to prevent blind signing vulnerabilities 
in cryptocurrency transactions.
"""

__version__ = "1.0.0"
__author__ = "ZKML-Guard Contributors"
__license__ = "MIT"

from src.model.zkml_guard_model import ZKMLGuardModel
from src.feature_extraction.transaction_features import TransactionFeatureExtractor
from src.zkml.zkml_proof import ProofGenerator, ProofVerifier

__all__ = [
    "ZKMLGuardModel",
    "TransactionFeatureExtractor", 
    "ProofGenerator",
    "ProofVerifier"
]
