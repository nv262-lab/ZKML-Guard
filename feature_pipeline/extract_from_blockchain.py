"""Extract features from blockchain data."""
from web3 import Web3
from src.feature_extraction.transaction_features import TransactionFeatureExtractor
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tx-hash', required=True)
    parser.add_argument('--network', default='ethereum')
    args = parser.parse_args()
    
    # Connect to network
    w3 = Web3(Web3.HTTPProvider(f'https://{args.network}.infura.io'))
    
    # Get transaction
    tx = w3.eth.get_transaction(args.tx_hash)
    
    # Extract features
    extractor = TransactionFeatureExtractor(w3)
    features = extractor.extract_features(dict(tx))
    
    print(f"Extracted features: {features.shape}")
    
if __name__ == "__main__":
    main()
