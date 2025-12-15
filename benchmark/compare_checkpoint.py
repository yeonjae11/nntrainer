#!/usr/bin/env python3
"""
Compare memory usage and loss between normal and checkpoint versions of SwiGLU FFN model.
"""

import subprocess
import re
import sys
from typing import Tuple, Optional

# Paths to executables
NORMAL_EXEC = "/home/duswo1120/nntrainer/build/Applications/SwiGLU_FFN/jni/nntrainer_myapp"
CHECKPOINT_EXEC = "/home/duswo1120/nntrainer/build/Applications/SwiGLU_FFN_Checkpoint/jni/nntrainer_swiglu_checkpoint"

# Default parameters
DEFAULT_SEQ_LEN = 128
DEFAULT_MODEL_DIM = 256
DEFAULT_NUM_HEADS = 8
DEFAULT_FFN_DIM = 1024
DEFAULT_EPOCHS = 1


def run_model(executable: str, batch_size: int, num_layers: int, 
              seq_len: int = DEFAULT_SEQ_LEN, model_dim: int = DEFAULT_MODEL_DIM,
              num_heads: int = DEFAULT_NUM_HEADS, ffn_dim: int = DEFAULT_FFN_DIM,
              epochs: int = DEFAULT_EPOCHS) -> Tuple[Optional[float], Optional[int]]:
    """
    Run the model and extract loss and peak memory.
    
    Returns:
        Tuple of (loss, peak_memory_mb)
    """
    cmd = [
        executable,
        "--batch_size", str(batch_size),
        "--num_layers", str(num_layers),
        "--seq_len", str(seq_len),
        "--model_dim", str(model_dim),
        "--num_heads", str(num_heads),
        "--ffn_dim", str(ffn_dim),
        "--epochs", str(epochs)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        output = result.stdout + result.stderr
        
        # Extract loss
        loss_match = re.search(r'Training Loss:\s+([\d.]+)', output)
        loss = float(loss_match.group(1)) if loss_match else None
        
        # Extract peak memory
        memory_match = re.search(r'After Training.*?Peak:\s+(\d+)\s+MB', output, re.DOTALL)
        peak_memory = int(memory_match.group(1)) if memory_match else None
        
        if loss is None:
            print(f"\nWarning: Could not extract loss from output", file=sys.stderr)
        if peak_memory is None:
            print(f"\nWarning: Could not extract memory from output", file=sys.stderr)
        
        return loss, peak_memory
    except subprocess.TimeoutExpired:
        print(f"\nTimeout running {executable}", file=sys.stderr)
        return None, None
    except Exception as e:
        print(f"\nError running {executable}: {e}", file=sys.stderr)
        return None, None


def compare_configurations(layer_counts: list, batch_sizes: list):
    """
    Compare normal and checkpoint versions across different configurations.
    """
    print("=" * 100)
    print("CHECKPOINT vs NORMAL COMPARISON")
    print("=" * 100)
    print()
    
    results = []
    
    for num_layers in layer_counts:
        for batch_size in batch_sizes:
            print(f"Testing: Layers={num_layers}, Batch={batch_size}...", end=" ", flush=True)
            
            # Run normal version
            normal_loss, normal_memory = run_model(NORMAL_EXEC, batch_size, num_layers)
            
            # Run checkpoint version
            checkpoint_loss, checkpoint_memory = run_model(CHECKPOINT_EXEC, batch_size, num_layers)
            
            if normal_loss is None or checkpoint_loss is None or normal_memory is None or checkpoint_memory is None:
                print("FAILED")
                exit(1)
                continue
            
            # Calculate savings
            memory_savings = normal_memory - checkpoint_memory
            memory_savings_pct = (memory_savings / normal_memory * 100) if normal_memory > 0 else 0
            
            # Check if losses match
            loss_match = abs(normal_loss - checkpoint_loss) < 1e-5
            if not loss_match:
                print(f"\nLoss mismatch: Normal={normal_loss}, Checkpoint={checkpoint_loss}")
                exit(1)
            
            results.append({
                'layers': num_layers,
                'batch': batch_size,
                'normal_loss': normal_loss,
                'checkpoint_loss': checkpoint_loss,
                'loss_match': loss_match,
                'normal_memory': normal_memory,
                'checkpoint_memory': checkpoint_memory,
                'memory_savings': memory_savings,
                'memory_savings_pct': memory_savings_pct
            })
            
            print("DONE")
    
    # Print results table
    print()
    print("=" * 100)
    print("RESULTS")
    print("=" * 100)
    print()
    
    # Memory comparison
    print("MEMORY COMPARISON (Peak MB)")
    print("-" * 100)
    print(f"{'Layers':<8} {'Batch':<8} {'Normal':<12} {'Checkpoint':<12} {'Savings':<12} {'Savings %':<12}")
    print("-" * 100)
    
    for r in results:
        print(f"{r['layers']:<8} {r['batch']:<8} {r['normal_memory']:<12} "
              f"{r['checkpoint_memory']:<12} {r['memory_savings']:<12} "
              f"{r['memory_savings_pct']:<12.1f}%")
    
    print()
    
    # Loss comparison
    print("LOSS COMPARISON")
    print("-" * 100)
    print(f"{'Layers':<8} {'Batch':<8} {'Normal':<15} {'Checkpoint':<15} {'Match':<8}")
    print("-" * 100)
    
    for r in results:
        match_str = "✅" if r['loss_match'] else "❌"
        print(f"{r['layers']:<8} {r['batch']:<8} {r['normal_loss']:<15.5f} "
              f"{r['checkpoint_loss']:<15.5f} {match_str:<8}")
    
    print()
    
    # Summary
    print("SUMMARY")
    print("-" * 100)
    total_tests = len(results)
    loss_matches = sum(1 for r in results if r['loss_match'])
    avg_savings = sum(r['memory_savings_pct'] for r in results) / total_tests if total_tests > 0 else 0
    max_savings = max((r['memory_savings_pct'] for r in results), default=0)
    
    print(f"Total tests: {total_tests}")
    if total_tests > 0:
        print(f"Loss matches: {loss_matches}/{total_tests} ({loss_matches/total_tests*100:.1f}%)")
        print(f"Average memory savings: {avg_savings:.1f}%")
        print(f"Maximum memory savings: {max_savings:.1f}%")
    else:
        print("No successful tests completed")
    print()


if __name__ == "__main__":
    # Default configurations to test
    layer_counts = [1, 2, 4, 6]
    batch_sizes = [4, 8]
    
    # Allow command line override
    if len(sys.argv) > 1:
        layer_counts = [int(x) for x in sys.argv[1].split(',')]
    if len(sys.argv) > 2:
        batch_sizes = [int(x) for x in sys.argv[2].split(',')]
    
    print(f"Testing layer counts: {layer_counts}")
    print(f"Testing batch sizes: {batch_sizes}")
    print()
    
    compare_configurations(layer_counts, batch_sizes)
