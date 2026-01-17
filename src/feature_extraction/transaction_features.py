"""
Transaction Feature Extraction for ZKML-Guard

This module extracts 42 features from blockchain transactions for risk assessment:
- Reputation features (12): Address history and behavior
- Calldata features (10): Smart contract interaction patterns
- Value distribution features (8): Transaction amounts and gas
- Temporal features (6): Timing and sequencing
- Smart contract features (6): Contract verification and patterns
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from web3 import Web3
from eth_utils import to_checksum_address
import json


class TransactionFeatureExtractor:
    """
    Extract features from blockchain transactions for ZKML-Guard classification.
    """
    
    def __init__(self, web3_provider: Optional[str] = None):
        """
        Initialize feature extractor.
        
        Args:
            web3_provider: Optional Web3 provider URL for on-chain data
        """
        self.w3 = Web3(Web3.HTTPProvider(web3_provider)) if web3_provider else None
        
        # Known malicious addresses (example - load from database in production)
        self.malicious_addresses = set()
        
        # Address reputation cache
        self.address_cache = {}
    
    def extract_features(self, transaction: Dict) -> np.ndarray:
        """
        Extract all features from a transaction.
        
        Args:
            transaction: Transaction dictionary with keys:
                - 'to': Destination address
                - 'from': Source address
                - 'value': Transaction value in wei
                - 'data': Transaction calldata
                - 'gas': Gas limit
                - 'gasPrice': Gas price
                - 'timestamp': Transaction timestamp
                - 'nonce': Transaction nonce
                
        Returns:
            Feature vector of shape (42,)
        """
        features = []
        
        # Extract feature groups
        features.extend(self._extract_reputation_features(transaction))
        features.extend(self._extract_calldata_features(transaction))
        features.extend(self._extract_value_features(transaction))
        features.extend(self._extract_temporal_features(transaction))
        features.extend(self._extract_contract_features(transaction))
        
        return np.array(features, dtype=np.float32)
    
    def _extract_reputation_features(self, tx: Dict) -> List[float]:
        """
        Extract reputation features (12 features).
        
        Features:
        1. Destination address age (days)
        2. Total transaction count for destination
        3. Interaction diversity (unique senders)
        4. Known malicious flag (0/1)
        5. Sanctioned address flag (0/1)
        6. Contract deployment flag (0/1)
        7. Source address age (days)
        8. Source total transactions
        9. Source interaction diversity
        10. Previous interactions between sender/receiver
        11. Destination balance (ETH)
        12. Source balance (ETH)
        """
        features = []
        
        to_addr = tx.get('to', '').lower()
        from_addr = tx.get('from', '').lower()
        
        # Destination features (1-6)
        dest_age_days = self._get_address_age(to_addr)
        features.append(dest_age_days)
        
        dest_tx_count = self._get_transaction_count(to_addr)
        features.append(np.log1p(dest_tx_count))  # Log-scale
        
        dest_diversity = self._get_interaction_diversity(to_addr)
        features.append(np.log1p(dest_diversity))
        
        features.append(1.0 if to_addr in self.malicious_addresses else 0.0)
        features.append(self._check_sanctioned(to_addr))
        features.append(self._is_contract(to_addr))
        
        # Source features (7-10)
        source_age_days = self._get_address_age(from_addr)
        features.append(source_age_days)
        
        source_tx_count = self._get_transaction_count(from_addr)
        features.append(np.log1p(source_tx_count))
        
        source_diversity = self._get_interaction_diversity(from_addr)
        features.append(np.log1p(source_diversity))
        
        prev_interactions = self._get_previous_interactions(from_addr, to_addr)
        features.append(np.log1p(prev_interactions))
        
        # Balance features (11-12)
        dest_balance = self._get_balance(to_addr)
        features.append(np.log1p(dest_balance))
        
        source_balance = self._get_balance(from_addr)
        features.append(np.log1p(source_balance))
        
        return features
    
    def _extract_calldata_features(self, tx: Dict) -> List[float]:
        """
        Extract calldata pattern features (10 features).
        
        Features:
        1. Calldata length (bytes)
        2. Function selector (normalized)
        3. Number of parameters
        4. Presence of delegatecall (0/1)
        5. Nested call depth
        6. Data complexity score
        7. Unusual patterns flag (0/1)
        8. Parameter type diversity
        9. Data entropy
        10. Known safe function flag (0/1)
        """
        features = []
        
        calldata = tx.get('data', '0x')
        
        # Calldata length (1)
        calldata_len = len(calldata) - 2  # Remove '0x' prefix
        features.append(np.log1p(calldata_len / 2))  # Convert hex to bytes
        
        # Function selector (2)
        if len(calldata) >= 10:
            selector = calldata[2:10]
            selector_int = int(selector, 16) / (2**32)  # Normalize to [0, 1]
            features.append(selector_int)
        else:
            features.append(0.0)
        
        # Parameter analysis (3-4)
        num_params = self._count_parameters(calldata)
        features.append(num_params)
        
        has_delegatecall = self._detect_delegatecall(calldata)
        features.append(1.0 if has_delegatecall else 0.0)
        
        # Call depth and complexity (5-7)
        call_depth = self._estimate_call_depth(calldata)
        features.append(call_depth)
        
        complexity_score = self._compute_complexity_score(calldata)
        features.append(complexity_score)
        
        has_unusual_patterns = self._detect_unusual_patterns(calldata)
        features.append(1.0 if has_unusual_patterns else 0.0)
        
        # Parameter type diversity (8)
        type_diversity = self._compute_type_diversity(calldata)
        features.append(type_diversity)
        
        # Data entropy (9)
        entropy = self._compute_entropy(calldata)
        features.append(entropy)
        
        # Known safe function (10)
        is_safe_function = self._check_safe_function(calldata)
        features.append(1.0 if is_safe_function else 0.0)
        
        return features
    
    def _extract_value_features(self, tx: Dict) -> List[float]:
        """
        Extract value distribution features (8 features).
        
        Features:
        1. Transaction value (ETH, log-scaled)
        2. Value as % of sender balance
        3. Value as % of receiver balance
        4. Gas limit (log-scaled)
        5. Gas price (Gwei, log-scaled)
        6. Total fee (ETH, log-scaled)
        7. Unusual gas limit flag (0/1)
        8. Priority fee ratio
        """
        features = []
        
        value_wei = int(tx.get('value', 0))
        value_eth = value_wei / 1e18
        features.append(np.log1p(value_eth))
        
        # Value as % of balances (2-3)
        from_balance = self._get_balance(tx.get('from', ''))
        to_balance = self._get_balance(tx.get('to', ''))
        
        value_pct_sender = value_eth / max(from_balance, 1e-9)
        features.append(min(value_pct_sender, 1.0))
        
        value_pct_receiver = value_eth / max(to_balance, 1e-9)
        features.append(min(value_pct_receiver, 1.0))
        
        # Gas features (4-6)
        gas_limit = int(tx.get('gas', 21000))
        features.append(np.log1p(gas_limit))
        
        gas_price_wei = int(tx.get('gasPrice', 0))
        gas_price_gwei = gas_price_wei / 1e9
        features.append(np.log1p(gas_price_gwei))
        
        total_fee_eth = (gas_limit * gas_price_wei) / 1e18
        features.append(np.log1p(total_fee_eth))
        
        # Unusual gas (7)
        is_unusual_gas = self._check_unusual_gas(gas_limit, gas_price_wei)
        features.append(1.0 if is_unusual_gas else 0.0)
        
        # Priority fee ratio (8)
        max_priority_fee = int(tx.get('maxPriorityFeePerGas', 0))
        priority_ratio = max_priority_fee / max(gas_price_wei, 1)
        features.append(min(priority_ratio, 1.0))
        
        return features
    
    def _extract_temporal_features(self, tx: Dict) -> List[float]:
        """
        Extract temporal features (6 features).
        
        Features:
        1. Hour of day (0-23, normalized)
        2. Day of week (0-6, normalized)
        3. Time since last transaction from sender (minutes, log-scaled)
        4. Transaction burst indicator (0/1)
        5. Time deviation from sender's pattern
        6. Nonce gap indicator (0/1)
        """
        features = []
        
        timestamp = tx.get('timestamp', datetime.now().timestamp())
        dt = datetime.fromtimestamp(timestamp)
        
        # Time of day features (1-2)
        hour_normalized = dt.hour / 23.0
        features.append(hour_normalized)
        
        day_normalized = dt.weekday() / 6.0
        features.append(day_normalized)
        
        # Time since last tx (3)
        from_addr = tx.get('from', '')
        time_since_last = self._time_since_last_tx(from_addr, timestamp)
        features.append(np.log1p(time_since_last / 60))  # Convert to minutes
        
        # Burst indicator (4)
        is_burst = 1.0 if time_since_last < 60 else 0.0  # < 1 minute
        features.append(is_burst)
        
        # Time deviation (5)
        time_deviation = self._compute_time_deviation(from_addr, dt)
        features.append(time_deviation)
        
        # Nonce gap (6)
        expected_nonce = self._get_transaction_count(from_addr)
        actual_nonce = tx.get('nonce', expected_nonce)
        has_nonce_gap = 1.0 if actual_nonce > expected_nonce else 0.0
        features.append(has_nonce_gap)
        
        return features
    
    def _extract_contract_features(self, tx: Dict) -> List[float]:
        """
        Extract smart contract features (6 features).
        
        Features:
        1. Contract verified flag (0/1)
        2. Contract age (days, log-scaled)
        3. Proxy pattern indicator (0/1)
        4. Recent contract modification flag (0/1)
        5. Contract complexity score
        6. Known vulnerability flag (0/1)
        """
        features = []
        
        to_addr = tx.get('to', '')
        
        if not self._is_contract(to_addr):
            # Not a contract - return zeros
            return [0.0] * 6
        
        # Contract verification (1)
        is_verified = self._is_contract_verified(to_addr)
        features.append(1.0 if is_verified else 0.0)
        
        # Contract age (2)
        contract_age = self._get_contract_age(to_addr)
        features.append(np.log1p(contract_age))
        
        # Proxy pattern (3)
        is_proxy = self._detect_proxy_pattern(to_addr)
        features.append(1.0 if is_proxy else 0.0)
        
        # Recent modification (4)
        has_recent_mod = self._check_recent_modification(to_addr)
        features.append(1.0 if has_recent_mod else 0.0)
        
        # Complexity score (5)
        complexity = self._compute_contract_complexity(to_addr)
        features.append(complexity)
        
        # Known vulnerability (6)
        has_vuln = self._check_known_vulnerabilities(to_addr)
        features.append(1.0 if has_vuln else 0.0)
        
        return features
    
    # Helper methods (simplified - would use database/RPC in production)
    
    def _get_address_age(self, address: str) -> float:
        """Get address age in days."""
        # Simplified - would query first transaction timestamp
        return np.random.uniform(0, 1000) if address else 0.0
    
    def _get_transaction_count(self, address: str) -> int:
        """Get total transaction count for address."""
        if self.w3 and address:
            try:
                return self.w3.eth.get_transaction_count(to_checksum_address(address))
            except:
                pass
        return int(np.random.uniform(0, 10000))
    
    def _get_interaction_diversity(self, address: str) -> int:
        """Get number of unique addresses interacted with."""
        return int(np.random.uniform(0, 500))
    
    def _check_sanctioned(self, address: str) -> float:
        """Check if address is sanctioned."""
        # Would check against OFAC/sanctions lists
        return 0.0
    
    def _is_contract(self, address: str) -> float:
        """Check if address is a contract."""
        if self.w3 and address:
            try:
                code = self.w3.eth.get_code(to_checksum_address(address))
                return 1.0 if len(code) > 2 else 0.0
            except:
                pass
        return 0.0
    
    def _get_balance(self, address: str) -> float:
        """Get address balance in ETH."""
        if self.w3 and address:
            try:
                balance_wei = self.w3.eth.get_balance(to_checksum_address(address))
                return balance_wei / 1e18
            except:
                pass
        return 0.0
    
    def _get_previous_interactions(self, addr1: str, addr2: str) -> int:
        """Count previous interactions between two addresses."""
        return int(np.random.uniform(0, 50))
    
    def _count_parameters(self, calldata: str) -> int:
        """Estimate number of parameters in calldata."""
        if len(calldata) <= 10:
            return 0
        param_data = calldata[10:]  # Skip function selector
        return len(param_data) // 64  # Each parameter is 32 bytes = 64 hex chars
    
    def _detect_delegatecall(self, calldata: str) -> bool:
        """Detect delegatecall opcode in calldata."""
        # Simplified - would analyze bytecode
        delegatecall_selector = 'f4'  # delegatecall opcode
        return delegatecall_selector in calldata.lower()
    
    def _estimate_call_depth(self, calldata: str) -> float:
        """Estimate nested call depth."""
        # Simplified heuristic
        return min(calldata.count('00') / 100, 5.0)
    
    def _compute_complexity_score(self, calldata: str) -> float:
        """Compute calldata complexity score."""
        if len(calldata) <= 2:
            return 0.0
        
        # Simple entropy-based complexity
        data = calldata[2:]  # Remove '0x'
        byte_counts = {}
        for i in range(0, len(data), 2):
            byte = data[i:i+2]
            byte_counts[byte] = byte_counts.get(byte, 0) + 1
        
        total_bytes = len(data) // 2
        entropy = 0
        for count in byte_counts.values():
            p = count / total_bytes
            entropy -= p * np.log2(p) if p > 0 else 0
        
        return entropy / 8.0  # Normalize to [0, 1]
    
    def _detect_unusual_patterns(self, calldata: str) -> bool:
        """Detect unusual patterns in calldata."""
        # Simplified - would use pattern matching
        return len(calldata) > 10000  # Very long calldata is suspicious
    
    def _compute_type_diversity(self, calldata: str) -> float:
        """Compute diversity of parameter types."""
        return np.random.uniform(0, 1)
    
    def _compute_entropy(self, calldata: str) -> float:
        """Compute Shannon entropy of calldata."""
        if len(calldata) <= 2:
            return 0.0
        
        data = calldata[2:]
        byte_counts = {}
        for i in range(0, len(data), 2):
            byte = data[i:i+2]
            byte_counts[byte] = byte_counts.get(byte, 0) + 1
        
        total = len(data) // 2
        entropy = -sum((c/total) * np.log2(c/total) for c in byte_counts.values())
        return entropy / 8.0
    
    def _check_safe_function(self, calldata: str) -> bool:
        """Check if function selector is known safe."""
        safe_selectors = {
            'a9059cbb',  # transfer
            '095ea7b3',  # approve
            '23b872dd',  # transferFrom
        }
        if len(calldata) >= 10:
            selector = calldata[2:10].lower()
            return selector in safe_selectors
        return False
    
    def _check_unusual_gas(self, gas_limit: int, gas_price: int) -> bool:
        """Check for unusual gas settings."""
        # Very high gas limit or very high gas price
        return gas_limit > 10000000 or gas_price > 500e9
    
    def _time_since_last_tx(self, address: str, current_time: float) -> float:
        """Get time since last transaction in seconds."""
        return np.random.uniform(60, 86400)  # 1 min to 1 day
    
    def _compute_time_deviation(self, address: str, dt: datetime) -> float:
        """Compute deviation from address's typical transaction times."""
        return np.random.uniform(0, 1)
    
    def _is_contract_verified(self, address: str) -> bool:
        """Check if contract is verified on Etherscan."""
        return np.random.random() > 0.5
    
    def _get_contract_age(self, address: str) -> float:
        """Get contract age in days."""
        return np.random.uniform(0, 1000)
    
    def _detect_proxy_pattern(self, address: str) -> bool:
        """Detect if contract uses proxy pattern."""
        return np.random.random() > 0.8
    
    def _check_recent_modification(self, address: str) -> bool:
        """Check if contract was recently modified."""
        return np.random.random() > 0.9
    
    def _compute_contract_complexity(self, address: str) -> float:
        """Compute contract complexity score."""
        return np.random.uniform(0, 1)
    
    def _check_known_vulnerabilities(self, address: str) -> bool:
        """Check for known vulnerabilities."""
        return False  # Would check vulnerability database


if __name__ == "__main__":
    # Example usage
    extractor = TransactionFeatureExtractor()
    
    # Example transaction
    tx = {
        'to': '0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb',
        'from': '0x123456789abcdef123456789abcdef123456789a',
        'value': '1000000000000000000',  # 1 ETH
        'data': '0xa9059cbb000000000000000000000000742d35cc6634c0532925a3b844bc9e7595f0beb0000000000000000000000000000000000000000000000000de0b6b3a7640000',
        'gas': '21000',
        'gasPrice': '50000000000',  # 50 Gwei
        'timestamp': datetime.now().timestamp(),
        'nonce': 0
    }
    
    features = extractor.extract_features(tx)
    print(f"Extracted {len(features)} features:")
    print(features)
