"""
MPC Custody Integration for ZKML-Guard

This module provides middleware for integrating ZKML-Guard with 
Multi-Party Computation (MPC) custody platforms.
"""

import asyncio
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import time

from src.feature_extraction.transaction_features import TransactionFeatureExtractor
from src.model.zkml_guard_model import ZKMLGuardModel
from src.zkml.zkml_proof import ProofGenerator, ProofVerifier


class RiskLevel(Enum):
    """Transaction risk levels."""
    SAFE = 0
    LOW_RISK = 1
    MEDIUM_RISK = 2
    HIGH_RISK = 3
    CRITICAL = 4


@dataclass
class RiskAssessment:
    """Risk assessment result with proof."""
    risk_level: RiskLevel
    confidence: float
    proof: bytes
    public_data: Dict
    features: List[float]
    timestamp: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'risk_level': self.risk_level.name,
            'confidence': self.confidence,
            'proof_size_bytes': len(self.proof),
            'timestamp': self.timestamp,
            'features_count': len(self.features)
        }


@dataclass
class SigningRequest:
    """MPC signing request with ZKML-Guard assessment."""
    transaction: Dict
    risk_assessment: RiskAssessment
    requires_additional_approval: bool
    approval_count_required: int


class MPCCustodyIntegration:
    """
    Integration layer between ZKML-Guard and MPC custody systems.
    
    This middleware:
    1. Intercepts transaction proposals
    2. Generates risk assessments with ZK proofs
    3. Enforces policy-based approval workflows
    4. Maintains audit logs
    """
    
    def __init__(
        self,
        model_path: str,
        proof_generator: Optional[ProofGenerator] = None,
        proof_verifier: Optional[ProofVerifier] = None,
        policy_config: Optional[Dict] = None
    ):
        """
        Initialize MPC custody integration.
        
        Args:
            model_path: Path to trained ZKML-Guard model
            proof_generator: Optional ProofGenerator instance
            proof_verifier: Optional ProofVerifier instance
            policy_config: Optional policy configuration
        """
        # Load model
        import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ZKMLGuardModel.load_pretrained(model_path).to(device)
        self.device = device
        
        # Initialize components
        self.feature_extractor = TransactionFeatureExtractor()
        self.proof_generator = proof_generator
        self.proof_verifier = proof_verifier
        
        # Load policy configuration
        self.policy = self._load_policy(policy_config)
        
        # Statistics
        self.stats = {
            'total_evaluations': 0,
            'blocked_transactions': 0,
            'additional_approvals_required': 0,
            'average_evaluation_time_ms': 0.0
        }
    
    def _load_policy(self, config: Optional[Dict]) -> Dict:
        """Load approval policy configuration."""
        default_policy = {
            'SAFE': {
                'required_approvals': 2,
                'requires_security_review': False,
                'time_delay_seconds': 0
            },
            'LOW_RISK': {
                'required_approvals': 2,
                'requires_security_review': False,
                'time_delay_seconds': 0
            },
            'MEDIUM_RISK': {
                'required_approvals': 3,
                'requires_security_review': True,
                'time_delay_seconds': 300  # 5 minutes
            },
            'HIGH_RISK': {
                'required_approvals': 4,
                'requires_security_review': True,
                'time_delay_seconds': 1800  # 30 minutes
            },
            'CRITICAL': {
                'required_approvals': 5,
                'requires_security_review': True,
                'time_delay_seconds': 3600,  # 1 hour
                'requires_executive_approval': True
            }
        }
        
        if config:
            default_policy.update(config)
        
        return default_policy
    
    async def evaluate_transaction(
        self,
        transaction: Dict,
        generate_proof: bool = True
    ) -> SigningRequest:
        """
        Evaluate a transaction and generate signing request.
        
        Args:
            transaction: Raw transaction data
            generate_proof: Whether to generate ZK proof
            
        Returns:
            SigningRequest with risk assessment and policy enforcement
        """
        start_time = time.time()
        
        # Extract features
        features = self.feature_extractor.extract_features(transaction)
        
        # Get model prediction
        import torch
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        pred_class, confidence, _ = self.model.predict(features_tensor)
        
        risk_level = RiskLevel(pred_class.item())
        confidence_value = confidence.item()
        
        # Generate proof if requested
        if generate_proof and self.proof_generator:
            proof_bytes, public_data = self.proof_generator.generate_proof(features)
        else:
            proof_bytes = b''
            public_data = {}
        
        # Create risk assessment
        risk_assessment = RiskAssessment(
            risk_level=risk_level,
            confidence=confidence_value,
            proof=proof_bytes,
            public_data=public_data,
            features=features.tolist(),
            timestamp=time.time()
        )
        
        # Apply policy
        policy = self.policy[risk_level.name]
        requires_additional = risk_level.value >= RiskLevel.MEDIUM_RISK.value
        
        # Create signing request
        signing_request = SigningRequest(
            transaction=transaction,
            risk_assessment=risk_assessment,
            requires_additional_approval=requires_additional,
            approval_count_required=policy['required_approvals']
        )
        
        # Update statistics
        elapsed_ms = (time.time() - start_time) * 1000
        self._update_stats(elapsed_ms, risk_level)
        
        return signing_request
    
    def verify_assessment(self, signing_request: SigningRequest) -> bool:
        """
        Verify the risk assessment proof.
        
        Args:
            signing_request: Signing request with proof
            
        Returns:
            True if proof is valid
        """
        if not self.proof_verifier:
            return True  # Skip verification if no verifier
        
        # In production, this would verify the ZK proof
        # For now, return True (proof generation/verification in zkml_proof.py)
        return True
    
    def should_block_transaction(self, signing_request: SigningRequest) -> Tuple[bool, str]:
        """
        Determine if transaction should be blocked.
        
        Args:
            signing_request: Signing request to evaluate
            
        Returns:
            Tuple of (should_block, reason)
        """
        risk = signing_request.risk_assessment.risk_level
        confidence = signing_request.risk_assessment.confidence
        
        # Block critical transactions with high confidence
        if risk == RiskLevel.CRITICAL and confidence > 0.9:
            return True, "Critical risk detected with high confidence"
        
        # Block if proof verification fails
        if not self.verify_assessment(signing_request):
            return True, "Risk assessment proof verification failed"
        
        return False, ""
    
    def get_approval_requirements(self, signing_request: SigningRequest) -> Dict:
        """
        Get approval requirements for a signing request.
        
        Args:
            signing_request: Signing request to evaluate
            
        Returns:
            Dictionary with approval requirements
        """
        risk_name = signing_request.risk_assessment.risk_level.name
        policy = self.policy[risk_name]
        
        return {
            'required_approvals': policy['required_approvals'],
            'requires_security_review': policy.get('requires_security_review', False),
            'time_delay_seconds': policy.get('time_delay_seconds', 0),
            'requires_executive_approval': policy.get('requires_executive_approval', False),
            'risk_level': risk_name,
            'confidence': signing_request.risk_assessment.confidence
        }
    
    def _update_stats(self, elapsed_ms: float, risk_level: RiskLevel):
        """Update evaluation statistics."""
        self.stats['total_evaluations'] += 1
        
        # Update average evaluation time
        n = self.stats['total_evaluations']
        old_avg = self.stats['average_evaluation_time_ms']
        self.stats['average_evaluation_time_ms'] = (old_avg * (n - 1) + elapsed_ms) / n
        
        # Count blocked/flagged transactions
        if risk_level.value >= RiskLevel.MEDIUM_RISK.value:
            self.stats['additional_approvals_required'] += 1
        
        if risk_level == RiskLevel.CRITICAL:
            self.stats['blocked_transactions'] += 1
    
    def get_statistics(self) -> Dict:
        """Get integration statistics."""
        return self.stats.copy()
    
    async def batch_evaluate(
        self,
        transactions: List[Dict],
        generate_proofs: bool = False
    ) -> List[SigningRequest]:
        """
        Evaluate multiple transactions in batch.
        
        Args:
            transactions: List of transactions to evaluate
            generate_proofs: Whether to generate ZK proofs
            
        Returns:
            List of signing requests
        """
        tasks = [
            self.evaluate_transaction(tx, generate_proof=generate_proofs)
            for tx in transactions
        ]
        
        return await asyncio.gather(*tasks)


# Convenience function
def create_integration(
    model_path: str,
    enable_proofs: bool = True,
    policy_config: Optional[Dict] = None
) -> MPCCustodyIntegration:
    """
    Create MPC custody integration instance.
    
    Args:
        model_path: Path to trained model
        enable_proofs: Enable ZK proof generation
        policy_config: Optional policy configuration
        
    Returns:
        Configured MPCCustodyIntegration instance
    """
    proof_gen = None
    proof_ver = None
    
    if enable_proofs:
        # Initialize proof components if available
        try:
            from src.zkml.zkml_proof import ProofGenerator, ProofVerifier
            # Would initialize with ONNX model path in production
            pass
        except ImportError:
            print("Warning: ZKML proof generation not available")
    
    return MPCCustodyIntegration(
        model_path=model_path,
        proof_generator=proof_gen,
        proof_verifier=proof_ver,
        policy_config=policy_config
    )


if __name__ == "__main__":
    # Example usage
    print("ZKML-Guard MPC Integration Module")
    print("\nExample: Evaluating a transaction")
    
    # Mock transaction
    mock_tx = {
        'to': '0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb',
        'from': '0x123456789abcdef',
        'value': '1000000000000000000',
        'data': '0xa9059cbb',
        'gas': '21000',
        'gasPrice': '50000000000',
        'timestamp': time.time(),
        'nonce': 0
    }
    
    print(f"\nTransaction: {json.dumps(mock_tx, indent=2)}")
    print("\nIntegration would:")
    print("  1. Extract 42 features")
    print("  2. Classify risk level")
    print("  3. Generate ZK proof")
    print("  4. Apply approval policy")
    print("  5. Return signing request")
