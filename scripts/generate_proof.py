#!/usr/bin/env python3
"""Standalone proof generation script."""
import argparse
import numpy as np
from src.zkml.zkml_proof import ProofGenerator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='ONNX model path')
    parser.add_argument('--features', required=True, help='Features .npy file')
    parser.add_argument('--output', default='proof.pf', help='Output proof file')
    args = parser.parse_args()
    
    features = np.load(args.features)
    generator = ProofGenerator(args.model)
    proof, public_data = generator.generate_proof(features)
    
    print(f"Proof generated: {len(proof)} bytes")
    
if __name__ == "__main__":
    main()
