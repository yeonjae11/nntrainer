#!/usr/bin/env python3
"""
Gradient Checkpointing Comparison Benchmark
Compares normal vs checkpoint training for GELU and SwiGLU FFN models
"""

import subprocess
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import os
import time
from datetime import datetime
from pathlib import Path

class GradientCheckpointingBenchmark:
    def __init__(self, gelu_exec: str, swiglu_exec: str):
        self.gelu_exec = gelu_exec
        self.swiglu_exec = swiglu_exec
        self.results = []
        
        # Create output directories
        self.base_dir = Path("gc_benchmark_results")
        self.logs_dir = self.base_dir / "logs"
        self.csv_dir = self.base_dir / "csv"
        self.plots_dir = self.base_dir / "plots"
        
        for dir_path in [self.logs_dir, self.csv_dir, self.plots_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        print(f"[INFO] Benchmark results will be saved to: {self.base_dir}")
    
    def run_experiment(self, executable: str, model_name: str, enable_checkpoint: bool, **kwargs) -> Optional[Dict]:
        """Run a single experiment with given hyperparameters"""
        # Build command line arguments
        cmd = [executable]
        for key, value in kwargs.items():
            cmd.extend([f"--{key}", str(value)])
        
        if enable_checkpoint:
            cmd.append("--enable-checkpoint")
        
        mode = "checkpoint" if enable_checkpoint else "normal"
        print(f"Running {model_name} ({mode}): {' '.join(cmd)}")
        
        # Generate log filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        param_str = "_".join([f"{k}{v}" for k, v in sorted(kwargs.items())])
        log_filename = self.logs_dir / f"{model_name}_{mode}_{timestamp}_{param_str}.log"
        
        try:
            # Run the experiment
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=None)
            
            # Save full log
            with open(log_filename, 'w') as f:
                f.write(f"Command: {' '.join(cmd)}\n")
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"Return code: {result.returncode}\n")
                f.write("\n=== STDOUT ===\n")
                f.write(result.stdout)
                f.write("\n=== STDERR ===\n")
                f.write(result.stderr)
            
            if result.returncode != 0:
                print(f"[ERROR] Experiment failed with return code {result.returncode}")
                print(f"[ERROR] Log saved to: {log_filename}")
                return None
            
            output = result.stdout + result.stderr
            
            # Parse results
            metrics = self.parse_output(output)
            metrics.update(kwargs)
            metrics['model'] = model_name
            metrics['checkpoint'] = enable_checkpoint
            metrics['log_file'] = str(log_filename)
            
            print(f"[SUCCESS] Log saved to: {log_filename}")
            return metrics
            
        except Exception as e:
            print(f"[ERROR] Exception during experiment: {e}")
            return None
    
    def parse_output(self, output: str) -> Dict:
        """Parse metrics from model output"""
        metrics = {
            'peak_memory_mb': 0,
            'rss_memory_mb': 0,
            'training_time_ms': 0,
            'final_loss': 0.0,
            'avg_total_latency_us': 0,
            'avg_forward_latency_us': 0,
            'avg_backward_latency_us': 0
        }
        
        # Parse iteration latencies and memory (RSS from [ITER])
        total_latencies = []
        forward_latencies = []
        backward_latencies = []
        memory_values = []
        
        for line in output.split('\n'):
            # Parse [ITER] output with memory: [ITER] #1 - Total: 1250 μs, Forward: 800 μs, Backward: 450 μs, Memory: 45 MB
            # This Memory value is RSS (Resident Set Size) - actual physical memory usage
            iter_match = re.search(r'\[ITER\] #\d+ - Total: (\d+) μs, Forward: (\d+) μs, Backward: (\d+) μs, Memory: (\d+) MB', line)
            if iter_match:
                total_latencies.append(int(iter_match.group(1)))
                forward_latencies.append(int(iter_match.group(2)))
                backward_latencies.append(int(iter_match.group(3)))
                memory_values.append(int(iter_match.group(4)))
            
            # Parse RSS from [MEMORY] output: [MEMORY] After Training - Peak: 4554 MB, Current: 4486 MB, RSS: 27 MB
            memory_match = re.search(r'\[MEMORY\].*?RSS:\s+(\d+)\s+MB', line)
            if memory_match:
                metrics['rss_memory_mb'] = max(metrics['rss_memory_mb'], int(memory_match.group(1)))
            
            # Parse training loss: Training Loss: 0.12345
            loss_match = re.search(r'Training Loss:\s+([\d.]+)', line)
            if loss_match:
                metrics['training_loss'] = float(loss_match.group(1))
        
        # Calculate latency statistics
        if total_latencies:
            metrics['avg_total_latency_us'] = np.mean(total_latencies)
            metrics['avg_forward_latency_us'] = np.mean(forward_latencies)
            metrics['avg_backward_latency_us'] = np.mean(backward_latencies)
        
        # Use maximum RSS from [ITER] outputs as peak memory (training중 최대 메모리)
        if memory_values:
            metrics['peak_memory_mb'] = max(memory_values)
        
        return metrics
    
    def compare_configuration(self, model_name: str, executable: str, **config) -> Optional[Dict]:
        """Compare normal vs checkpoint for a single configuration"""
        print(f"\n=== Testing {model_name}: {config} ===")
        
        # Run normal version
        print(f"[1/2] Running normal mode...")
        normal_metrics = self.run_experiment(executable, model_name, False, **config)
        if not normal_metrics:
            print(f"[FAILED] Normal mode failed")
            return None
        
        time.sleep(1)
        
        # Run checkpoint version
        print(f"[2/2] Running checkpoint mode...")
        checkpoint_metrics = self.run_experiment(executable, model_name, True, **config)
        if not checkpoint_metrics:
            print(f"[FAILED] Checkpoint mode failed")
            return None
        
        # Calculate comparisons
        result = {
            'model': model_name,
            **config,
            'normal_total_latency_us': normal_metrics['avg_total_latency_us'],
            'normal_forward_latency_us': normal_metrics['avg_forward_latency_us'],
            'normal_backward_latency_us': normal_metrics['avg_backward_latency_us'],
            'normal_memory_mb': normal_metrics['peak_memory_mb'],
            'normal_loss': normal_metrics['training_loss'],
            'checkpoint_total_latency_us': checkpoint_metrics['avg_total_latency_us'],
            'checkpoint_forward_latency_us': checkpoint_metrics['avg_forward_latency_us'],
            'checkpoint_backward_latency_us': checkpoint_metrics['avg_backward_latency_us'],
            'checkpoint_memory_mb': checkpoint_metrics['peak_memory_mb'],
            'checkpoint_loss': checkpoint_metrics['training_loss'],
        }
        
        # Calculate savings/overhead
        if normal_metrics['peak_memory_mb'] > 0:
            result['memory_savings_mb'] = normal_metrics['peak_memory_mb'] - checkpoint_metrics['peak_memory_mb']
            result['memory_savings_pct'] = (result['memory_savings_mb'] / normal_metrics['peak_memory_mb']) * 100
        
        if normal_metrics['avg_total_latency_us'] > 0:
            result['time_overhead_us'] = checkpoint_metrics['avg_total_latency_us'] - normal_metrics['avg_total_latency_us']
            result['time_overhead_pct'] = (result['time_overhead_us'] / normal_metrics['avg_total_latency_us']) * 100
        
        result['loss_diff'] = abs(normal_metrics['training_loss'] - checkpoint_metrics['training_loss'])
        result['loss_match'] = result['loss_diff'] < 1e-5
        
        print(f"[OK] Memory: {result['memory_savings_mb']:.0f} MB saved ({result['memory_savings_pct']:.1f}%)")
        print(f"[OK] Time: {result['time_overhead_us']:.0f} μs overhead ({result['time_overhead_pct']:.1f}%)")
        print(f"[OK] Loss: {'MATCH' if result['loss_match'] else 'MISMATCH'} (diff={result['loss_diff']:.8f})")
        
        return result
    
    def benchmark_layers(self, base_config: Dict) -> List[Dict]:
        """Benchmark different number of layers"""
        print("\n" + "="*80)
        print("Benchmarking Number of Layers")
        print("="*80)
        
        layer_counts = [64]  # Added: 2, 64
        results = []
        
        for num_layers in layer_counts:
            config = base_config.copy()
            config['num_layers'] = num_layers
            
            # Test GELU
            result = self.compare_configuration("GELU", self.gelu_exec, **config)
            if result:
                results.append(result)
            
            time.sleep(1)
            
            # Test SwiGLU
            result = self.compare_configuration("SwiGLU", self.swiglu_exec, **config)
            if result:
                results.append(result)
            
            time.sleep(1)
        
        return results

    def benchmark_seq_len(self, base_config: Dict) -> List[Dict]:
        """Benchmark different sequence lengths"""
        print("\n" + "="*80)
        print("Benchmarking Sequence Lengths")
        print("="*80)
        
        seq_lens = [1024]  # Added: 32, 1024
        results = []
        
        for seq_len in seq_lens:
            config = base_config.copy()
            config['seq_len'] = seq_len
            
            # Test GELU
            result = self.compare_configuration("GELU", self.gelu_exec, **config)
            if result:
                results.append(result)
            
            time.sleep(1)
            
            # Test SwiGLU
            result = self.compare_configuration("SwiGLU", self.swiglu_exec, **config)
            if result:
                results.append(result)
            
            time.sleep(1)
        
        return results
    
    def benchmark_model_scale(self, base_config: Dict) -> List[Dict]:
        """Benchmark different model scales (model_dim and ffn_dim together, maintaining 4x ratio)"""
        print("\n" + "="*80)
        print("Benchmarking Model Scale (model_dim : ffn_dim = 1:4)")
        print("="*80)
        
        # (model_dim, ffn_dim, num_heads) tuples
        scales = [
            # (64, 256, 2),     # XSmall - NEW
            # (128, 512, 4),    # Small - existing
            # (256, 1024, 8),   # Medium (base) - existing
            # (512, 2048, 8),   # Large - existing
            # (1024, 4096, 8),  # XLarge - existing
            (2048, 8192, 16), # XXLarge - NEW
        ]
        results = []
        
        for model_dim, ffn_dim, num_heads in scales:
            config = base_config.copy()
            config['model_dim'] = model_dim
            config['ffn_dim'] = ffn_dim
            config['num_heads'] = num_heads
            
            print(f"\n--- Scale: model_dim={model_dim}, ffn_dim={ffn_dim}, num_heads={num_heads} ---")
            
            # Test GELU
            result = self.compare_configuration("GELU", self.gelu_exec, **config)
            if result:
                results.append(result)
            
            time.sleep(1)
            
            # Test SwiGLU
            result = self.compare_configuration("SwiGLU", self.swiglu_exec, **config)
            if result:
                results.append(result)
            
            time.sleep(1)
        
        return results
    
    def benchmark_batch_sizes(self, base_config: Dict) -> List[Dict]:
        """Benchmark different batch sizes"""
        print("\n" + "="*80)
        print("Benchmarking Batch Sizes")
        print("="*80)
        
        batch_sizes = [1,2,4,8]
        results = []
        
        for batch_size in batch_sizes:
            config = base_config.copy()
            config['batch_size'] = batch_size
            
            # Test GELU
            result = self.compare_configuration("GELU", self.gelu_exec, **config)
            if result:
                results.append(result)
            
            time.sleep(1)
            
            # Test SwiGLU
            result = self.compare_configuration("SwiGLU", self.swiglu_exec, **config)
            if result:
                results.append(result)
            
            time.sleep(1)
        
        return results
    
    def plot_results(self, results: List[Dict]):
        """Create comparison plots"""
        if not results:
            print("[WARNING] No results to plot")
            return
        
        df = pd.DataFrame(results)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Gradient Checkpointing Performance Comparison', fontsize=16)
        
        # Group by model
        for model_name in df['model'].unique():
            model_df = df[df['model'] == model_name]
            
            # Plot 1: Memory Savings vs Layers
            if 'num_layers' in model_df.columns:
                ax = axes[0, 0]
                layers_df = model_df.groupby('num_layers').first()
                ax.plot(layers_df.index, layers_df['memory_savings_pct'], 'o-', label=model_name, linewidth=2, markersize=8)
                ax.set_xlabel('Number of Layers')
                ax.set_ylabel('Memory Savings (%)')
                ax.set_title('Memory Savings vs Layers')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            # Plot 2: Time Overhead vs Layers
            if 'num_layers' in model_df.columns:
                ax = axes[0, 1]
                layers_df = model_df.groupby('num_layers').first()
                ax.plot(layers_df.index, layers_df['time_overhead_pct'], 'o-', label=model_name, linewidth=2, markersize=8)
                ax.set_xlabel('Number of Layers')
                ax.set_ylabel('Time Overhead (%)')
                ax.set_title('Time Overhead vs Layers')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            # Plot 3: Forward/Backward Breakdown
            if 'num_layers' in model_df.columns:
                ax = axes[0, 2]
                layers_df = model_df.groupby('num_layers').first()
                fwd_overhead = ((layers_df['checkpoint_forward_latency_us'] - layers_df['normal_forward_latency_us']) / 
                               layers_df['normal_forward_latency_us'] * 100)
                bwd_overhead = ((layers_df['checkpoint_backward_latency_us'] - layers_df['normal_backward_latency_us']) / 
                               layers_df['normal_backward_latency_us'] * 100)
                ax.plot(layers_df.index, fwd_overhead, 's-', label=f'{model_name} Forward', linewidth=2, markersize=6)
                ax.plot(layers_df.index, bwd_overhead, '^-', label=f'{model_name} Backward', linewidth=2, markersize=6)
                ax.set_xlabel('Number of Layers')
                ax.set_ylabel('Overhead (%)')
                ax.set_title('Forward/Backward Overhead vs Layers')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            # Plot 4: Memory Savings vs Batch Size
            if 'batch_size' in model_df.columns:
                ax = axes[1, 0]
                batch_df = model_df.groupby('batch_size').first()
                ax.plot(batch_df.index, batch_df['memory_savings_pct'], 'o-', label=model_name, linewidth=2, markersize=8)
                ax.set_xlabel('Batch Size')
                ax.set_ylabel('Memory Savings (%)')
                ax.set_title('Memory Savings vs Batch Size')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            # Plot 5: Time Overhead vs Batch Size
            if 'batch_size' in model_df.columns:
                ax = axes[1, 1]
                batch_df = model_df.groupby('batch_size').first()
                ax.plot(batch_df.index, batch_df['time_overhead_pct'], 'o-', label=model_name, linewidth=2, markersize=8)
                ax.set_xlabel('Batch Size')
                ax.set_ylabel('Time Overhead (%)')
                ax.set_title('Time Overhead vs Batch Size')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            # Plot 6: Loss Comparison
            ax = axes[1, 2]
            ax.scatter(model_df['normal_loss'], model_df['checkpoint_loss'], alpha=0.6, s=100, label=model_name)
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
            ax.set_xlabel('Normal Loss')
            ax.set_ylabel('Checkpoint Loss')
            ax.set_title('Loss Comparison')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_filename = self.plots_dir / 'gc_comparison.png'
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n[SAVED] Plots: {plot_filename}")
    
    def save_results_csv(self, results: List[Dict]):
        """Save results to CSV"""
        if not results:
            print("[WARNING] No results to save")
            return
        
        try:
            df = pd.DataFrame(results)
            filename = self.csv_dir / 'gc_comparison_results.csv'
            df.to_csv(filename, index=False)
            print(f"[SAVED] CSV: {filename}")
        except Exception as e:
            print(f"[ERROR] Failed to save CSV: {e}")
    
    def print_summary(self, results: List[Dict]):
        """Print summary table"""
        if not results:
            print("[WARNING] No results to summarize")
            return
        
        print("\n" + "="*120)
        print("SUMMARY")
        print("="*120)
        
        df = pd.DataFrame(results)
        
        # Memory comparison
        print("\nMEMORY COMPARISON")
        print("-"*120)
        print(f"{'Model':<10} {'Layers':<8} {'Batch':<8} {'Normal (MB)':<12} {'Checkpoint (MB)':<15} {'Savings (MB)':<12} {'Savings %':<12}")
        print("-"*120)
        for _, row in df.iterrows():
            print(f"{row['model']:<10} {row.get('num_layers', '-'):<8} {row.get('batch_size', '-'):<8} "
                  f"{row['normal_memory_mb']:<12.0f} {row['checkpoint_memory_mb']:<15.0f} "
                  f"{row['memory_savings_mb']:<12.0f} {row['memory_savings_pct']:<12.1f}")
        
        # Timing comparison
        print("\nTIMING COMPARISON")
        print("-"*120)
        print(f"{'Model':<10} {'Layers':<8} {'Batch':<8} {'Normal (μs)':<12} {'Checkpoint (μs)':<15} {'Overhead (μs)':<14} {'Overhead %':<12}")
        print("-"*120)
        for _, row in df.iterrows():
            print(f"{row['model']:<10} {row.get('num_layers', '-'):<8} {row.get('batch_size', '-'):<8} "
                  f"{row['normal_total_latency_us']:<12.0f} {row['checkpoint_total_latency_us']:<15.0f} "
                  f"{row['time_overhead_us']:<14.0f} {row['time_overhead_pct']:<12.1f}")
        
        # Loss comparison
        print("\nLOSS COMPARISON")
        print("-"*120)
        print(f"{'Model':<10} {'Layers':<8} {'Batch':<8} {'Normal Loss':<15} {'Checkpoint Loss':<18} {'Diff':<15} {'Match':<8}")
        print("-"*120)
        for _, row in df.iterrows():
            match_str = "✅" if row['loss_match'] else "❌"
            print(f"{row['model']:<10} {row.get('num_layers', '-'):<8} {row.get('batch_size', '-'):<8} "
                  f"{row['normal_loss']:<15.5f} {row['checkpoint_loss']:<18.5f} "
                  f"{row['loss_diff']:<15.8f} {match_str:<8}")
        
        # Overall statistics
        print("\nOVERALL STATISTICS")
        print("-"*120)
        print(f"Total tests: {len(df)}")
        print(f"Loss matches: {df['loss_match'].sum()}/{len(df)} ({df['loss_match'].sum()/len(df)*100:.1f}%)")
        print(f"Average memory savings: {df['memory_savings_pct'].mean():.1f}%")
        print(f"Average time overhead: {df['time_overhead_pct'].mean():.1f}%")
        print("="*120)
    
    def run_full_benchmark(self):
        """Run complete benchmark suite"""
        base_config = {
            'seq_len': 64,
            'model_dim': 128,
            'num_heads': 4,
            'ffn_dim': 512,
            'epochs': 3,
            'learning_rate': 0.0001,
            'num_layers': 4,
            'batch_size': 4
        }
        
        print("Starting Gradient Checkpointing Comparison Benchmark...")
        print(f"Base configuration: {base_config}")
        
        all_results = []
        
        # Benchmark by layers
        # results = self.benchmark_layers(base_config)
        # all_results.extend(results)
        
        # # Benchmark by sequence length
        # results = self.benchmark_seq_len(base_config)
        # all_results.extend(results)
        
        # Benchmark by model scale (model_dim + ffn_dim together)
        results = self.benchmark_model_scale(base_config)
        all_results.extend(results)
        
        # Benchmark by batch size
        # results = self.benchmark_batch_sizes(base_config)
        # all_results.extend(results)
        
        # Save and visualize
        self.save_results_csv(all_results)
        self.plot_results(all_results)
        self.print_summary(all_results)
        
        return all_results

if __name__ == "__main__":
    # Find executables
    build_dir = Path("./nntrainer/build")
    gelu_exec = build_dir / "Applications/GC_Test/jni/gelu_gc_test"
    swiglu_exec = build_dir / "Applications/GC_Test/jni/swiglu_gc_test"
    
    # Check if executables exist
    if not gelu_exec.exists():
        print(f"[ERROR] GELU executable not found: {gelu_exec}")
        print("Please build first: ninja -C nntrainer/build")
        exit(1)
    
    if not swiglu_exec.exists():
        print(f"[ERROR] SwiGLU executable not found: {swiglu_exec}")
        print("Please build first: ninja -C nntrainer/build")
        exit(1)
    
    # Run benchmark
    benchmark = GradientCheckpointingBenchmark(str(gelu_exec), str(swiglu_exec))
    results = benchmark.run_full_benchmark()
    
    print("\n" + "="*80)
    print("Benchmark completed!")
    print(f"Results saved in: {benchmark.base_dir}")
    print("="*80)
