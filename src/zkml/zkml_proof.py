"""
Zero-Knowledge Proof Generation for ZKML-Guard

This module handles the generation of zkSNARK proofs for transaction
risk classification using EZKL framework.

Proof Generation Pipeline:
1. Load ONNX model and transaction features
2. Generate structured reference string (SRS)
3. Calibrate circuit for optimal quantization
4. Generate proof of correct inference
5. Output proof + public inputs/outputs
"""

import subprocess
import json
import time
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np
import onnx


class ProofGenerator:
    """
    Generate zero-knowledge proofs for ZKML-Guard model inference.
    
    Uses EZKL 12.0 framework with Halo2 proving system.
    """
    
    def __init__(
        self,
        model_path: str,
        srs_path: Optional[str] = None,
        compiled_circuit_path: Optional[str] = None,
        settings_path: Optional[str] = None
    ):
        """
        Initialize proof generator.
        
        Args:
            model_path: Path to ONNX model file
            srs_path: Path to structured reference string (generated if None)
            compiled_circuit_path: Path to compiled circuit (generated if None)
            settings_path: Path to EZKL settings (generated if None)
        """
        self.model_path = Path(model_path)
        self.srs_path = Path(srs_path) if srs_path else Path("srs.params")
        self.compiled_circuit_path = Path(compiled_circuit_path) if compiled_circuit_path else Path("circuit.compiled")
        self.settings_path = Path(settings_path) if settings_path else Path("settings.json")
        
        # Verification key path
        self.vk_path = Path("vk.key")
        self.pk_path = Path("pk.key")
        
        # Check if model exists
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        # Initialize if needed
        if not self.compiled_circuit_path.exists():
            print("Compiled circuit not found. Running setup...")
            self.setup()
    
    def setup(self, calibration_data: Optional[np.ndarray] = None):
        """
        Setup the ZKML circuit (one-time operation per model).
        
        This generates:
        - Settings file with quantization parameters
        - Structured Reference String (SRS)
        - Verification key (VK)
        - Proving key (PK)
        
        Args:
            calibration_data: Optional calibration data for quantization
                            (shape: [n_samples, n_features])
        """
        print("=== ZKML Circuit Setup ===")
        print(f"Model: {self.model_path}")
        
        # Step 1: Generate settings with calibration
        print("\n[1/5] Generating settings and calibrating...")
        
        if calibration_data is None:
            # Generate dummy calibration data
            calibration_data = np.random.randn(10, 42).astype(np.float32)
        
        # Save calibration data
        calib_path = Path("calibration.json")
        self._save_input_data(calibration_data, calib_path)
        
        # Generate settings
        cmd = [
            "ezkl", "gen-settings",
            "-M", str(self.model_path),
            "--settings-path", str(self.settings_path),
            "--input-visibility", "public",
            "--output-visibility", "public",
            "--param-visibility", "private"  # Keep model parameters private
        ]
        self._run_command(cmd, "Settings generation")
        
        # Calibrate settings
        cmd = [
            "ezkl", "calibrate-settings",
            "-M", str(self.model_path),
            "-D", str(calib_path),
            "--settings-path", str(self.settings_path),
            "--target", "resources"  # Optimize for resource usage
        ]
        self._run_command(cmd, "Settings calibration")
        
        # Step 2: Compile circuit
        print("\n[2/5] Compiling circuit...")
        cmd = [
            "ezkl", "compile-circuit",
            "-M", str(self.model_path),
            "-S", str(self.settings_path),
            "--compiled-circuit", str(self.compiled_circuit_path)
        ]
        self._run_command(cmd, "Circuit compilation")
        
        # Step 3: Get SRS (Structured Reference String)
        print("\n[3/5] Generating SRS...")
        start_time = time.time()
        cmd = [
            "ezkl", "get-srs",
            "-S", str(self.settings_path),
            "--srs-path", str(self.srs_path)
        ]
        self._run_command(cmd, "SRS generation")
        srs_time = time.time() - start_time
        print(f"SRS generation time: {srs_time:.2f}s")
        
        # Step 4: Setup (generate proving and verification keys)
        print("\n[4/5] Generating proving and verification keys...")
        start_time = time.time()
        cmd = [
            "ezkl", "setup",
            "-M", str(self.model_path),
            "--compiled-circuit", str(self.compiled_circuit_path),
            "--srs-path", str(self.srs_path),
            "--vk-path", str(self.vk_path),
            "--pk-path", str(self.pk_path)
        ]
        self._run_command(cmd, "Key generation")
        key_gen_time = time.time() - start_time
        print(f"Key generation time: {key_gen_time:.2f}s")
        
        # Step 5: Verify setup
        print("\n[5/5] Verifying setup...")
        self._verify_setup()
        
        print("\n=== Setup Complete ===")
        print(f"Verification key: {self.vk_path}")
        print(f"SRS: {self.srs_path}")
        print(f"Total setup time: {srs_time + key_gen_time:.2f}s")
    
    def generate_proof(
        self,
        features: np.ndarray,
        output_proof_path: Optional[str] = None,
        output_public_path: Optional[str] = None
    ) -> Tuple[bytes, Dict]:
        """
        Generate zero-knowledge proof for transaction classification.
        
        Args:
            features: Transaction features (shape: [n_features] or [1, n_features])
            output_proof_path: Optional path to save proof
            output_public_path: Optional path to save public inputs/outputs
            
        Returns:
            Tuple of (proof_bytes, public_data):
                - proof_bytes: Binary proof data
                - public_data: Dictionary with public inputs and outputs
        """
        # Ensure features are 2D
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Set default paths
        if output_proof_path is None:
            output_proof_path = "proof.pf"
        if output_public_path is None:
            output_public_path = "public.json"
        
        # Save input features
        input_path = Path("input.json")
        self._save_input_data(features, input_path)
        
        # Generate witness (intermediate values)
        print("Generating witness...")
        witness_path = Path("witness.json")
        cmd = [
            "ezkl", "gen-witness",
            "-M", str(self.model_path),
            "-D", str(input_path),
            "--output", str(witness_path),
            "--compiled-circuit", str(self.compiled_circuit_path)
        ]
        self._run_command(cmd, "Witness generation")
        
        # Generate proof
        print("Generating proof...")
        start_time = time.time()
        cmd = [
            "ezkl", "prove",
            "--witness", str(witness_path),
            "--compiled-circuit", str(self.compiled_circuit_path),
            "--pk-path", str(self.pk_path),
            "--proof-path", output_proof_path,
            "--srs-path", str(self.srs_path)
        ]
        self._run_command(cmd, "Proof generation")
        proof_time = time.time() - start_time
        
        print(f"Proof generation time: {proof_time * 1000:.0f}ms")
        
        # Read proof
        with open(output_proof_path, 'rb') as f:
            proof_bytes = f.read()
        
        # Extract public inputs/outputs from witness
        with open(witness_path, 'r') as f:
            witness_data = json.load(f)
        
        public_data = {
            'inputs': features.tolist(),
            'outputs': witness_data.get('outputs', []),
            'proof_size_bytes': len(proof_bytes),
            'generation_time_ms': proof_time * 1000
        }
        
        # Save public data
        with open(output_public_path, 'w') as f:
            json.dump(public_data, f, indent=2)
        
        return proof_bytes, public_data
    
    def _save_input_data(self, data: np.ndarray, filepath: Path):
        """Save input data in EZKL JSON format."""
        # Convert to list format expected by EZKL
        input_data = {
            "input_data": data.tolist()
        }
        with open(filepath, 'w') as f:
            json.dump(input_data, f)
    
    def _run_command(self, cmd: list, description: str):
        """Run EZKL command and handle errors."""
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            if result.stdout:
                print(f"  {description}: {result.stdout.strip()}")
        except subprocess.CalledProcessError as e:
            print(f"Error in {description}:")
            print(f"  Command: {' '.join(cmd)}")
            print(f"  Return code: {e.returncode}")
            print(f"  Stdout: {e.stdout}")
            print(f"  Stderr: {e.stderr}")
            raise
        except FileNotFoundError:
            print("Error: EZKL not found. Please install EZKL:")
            print("  curl https://github.com/zkonduit/ezkl/releases/download/v12.0.0/ezkl-linux-amd64 -L -o ezkl")
            print("  chmod +x ezkl")
            print("  sudo mv ezkl /usr/local/bin/")
            raise
    
    def _verify_setup(self):
        """Verify that setup was successful."""
        required_files = [
            self.settings_path,
            self.compiled_circuit_path,
            self.srs_path,
            self.vk_path,
            self.pk_path
        ]
        
        for file_path in required_files:
            if not file_path.exists():
                raise FileNotFoundError(f"Setup failed: {file_path} not found")
        
        print("Setup verification: All required files present ✓")


class ProofVerifier:
    """
    Verify zero-knowledge proofs for ZKML-Guard.
    """
    
    def __init__(self, vk_path: str, settings_path: Optional[str] = None):
        """
        Initialize proof verifier.
        
        Args:
            vk_path: Path to verification key
            settings_path: Path to settings file
        """
        self.vk_path = Path(vk_path)
        self.settings_path = Path(settings_path) if settings_path else Path("settings.json")
        
        if not self.vk_path.exists():
            raise FileNotFoundError(f"Verification key not found: {vk_path}")
    
    def verify(
        self,
        proof_path: str,
        public_data_path: str,
        srs_path: str = "srs.params"
    ) -> Tuple[bool, float]:
        """
        Verify a zero-knowledge proof.
        
        Args:
            proof_path: Path to proof file
            public_data_path: Path to public inputs/outputs
            srs_path: Path to SRS parameters
            
        Returns:
            Tuple of (is_valid, verification_time_ms)
        """
        print("Verifying proof...")
        start_time = time.time()
        
        cmd = [
            "ezkl", "verify",
            "--proof-path", proof_path,
            "--settings-path", str(self.settings_path),
            "--vk-path", str(self.vk_path),
            "--srs-path", srs_path
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            verification_time = (time.time() - start_time) * 1000
            
            # Check if verification succeeded
            is_valid = "verified" in result.stdout.lower()
            
            print(f"Verification result: {'✓ Valid' if is_valid else '✗ Invalid'}")
            print(f"Verification time: {verification_time:.2f}ms")
            
            return is_valid, verification_time
            
        except subprocess.CalledProcessError as e:
            print(f"Verification failed:")
            print(f"  Stderr: {e.stderr}")
            return False, (time.time() - start_time) * 1000


def benchmark_proof_generation(
    model_path: str,
    num_iterations: int = 100,
    features: Optional[np.ndarray] = None
) -> Dict:
    """
    Benchmark proof generation performance.
    
    Args:
        model_path: Path to ONNX model
        num_iterations: Number of proof generations to run
        features: Optional fixed features (generates random if None)
        
    Returns:
        Dictionary with benchmark statistics
    """
    print(f"=== Proof Generation Benchmark ({num_iterations} iterations) ===")
    
    # Initialize generator
    generator = ProofGenerator(model_path)
    
    if features is None:
        features = np.random.randn(1, 42).astype(np.float32)
    
    # Warmup
    print("Warming up...")
    generator.generate_proof(features)
    
    # Benchmark
    print(f"Running {num_iterations} iterations...")
    times = []
    proof_sizes = []
    
    for i in range(num_iterations):
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i + 1}/{num_iterations}")
        
        start = time.time()
        proof_bytes, public_data = generator.generate_proof(features)
        elapsed = (time.time() - start) * 1000
        
        times.append(elapsed)
        proof_sizes.append(len(proof_bytes))
    
    # Calculate statistics
    times = np.array(times)
    results = {
        'iterations': num_iterations,
        'median_ms': float(np.median(times)),
        'mean_ms': float(np.mean(times)),
        'std_ms': float(np.std(times)),
        'min_ms': float(np.min(times)),
        'max_ms': float(np.max(times)),
        'p95_ms': float(np.percentile(times, 95)),
        'p99_ms': float(np.percentile(times, 99)),
        'proof_size_bytes': proof_sizes[0],
        'proof_size_kb': proof_sizes[0] / 1024
    }
    
    print("\n=== Benchmark Results ===")
    print(f"Median: {results['median_ms']:.1f}ms")
    print(f"Mean: {results['mean_ms']:.1f}ms ± {results['std_ms']:.1f}ms")
    print(f"P95: {results['p95_ms']:.1f}ms")
    print(f"P99: {results['p99_ms']:.1f}ms")
    print(f"Range: [{results['min_ms']:.1f}, {results['max_ms']:.1f}]ms")
    print(f"Proof size: {results['proof_size_kb']:.2f}KB")
    
    return results


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python zkml_proof.py <model.onnx>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    
    # Initialize generator and run setup
    generator = ProofGenerator(model_path)
    
    # Generate example proof
    features = np.random.randn(42).astype(np.float32)
    proof, public_data = generator.generate_proof(features)
    
    print("\n=== Proof Generated ===")
    print(f"Proof size: {len(proof)} bytes")
    print(f"Public inputs: {len(public_data['inputs'])}")
    print(f"Public outputs: {public_data['outputs']}")
    
    # Verify proof
    verifier = ProofVerifier("vk.key")
    is_valid, verify_time = verifier.verify("proof.pf", "public.json")
    
    print(f"\nVerification: {'✓ VALID' if is_valid else '✗ INVALID'}")
    print(f"Verification time: {verify_time:.2f}ms")
