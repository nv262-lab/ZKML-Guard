#!/usr/bin/env python3
"""Benchmark proof generation performance."""
import argparse
import numpy as np
from src.zkml.zkml_proof import benchmark_proof_generation

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--iterations', type=int, default=100)
    args = parser.parse_args()
    
    results = benchmark_proof_generation(args.model, args.iterations)
    
    print(f"Median: {results['median_ms']:.1f}ms")
    print(f"P95: {results['p95_ms']:.1f}ms")
    
if __name__ == "__main__":
    main()
