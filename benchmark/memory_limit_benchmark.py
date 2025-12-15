#!/usr/bin/env python3
"""
Memory Limit Benchmark for Gradient Checkpointing
Tests maximum trainable model size under different memory constraints
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

class MemoryLimitBenchmark:
    def __init__(self, gelu_exec: str, swiglu_exec: str):
        self.gelu_exec = gelu_exec
        self.swiglu_exec = swiglu_exec
        
        # Create output directories
        self.base_dir = Path("memory_limit_results")
        self.logs_dir = self.base_dir / "logs"
        self.csv_dir = self.base_dir / "csv"
        self.plots_dir = self.base_dir / "plots"
        
        for dir_path in [self.logs_dir, self.csv_dir, self.plots_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        print(f"[INFO] Results will be saved to: {self.base_dir}")
    
    def calculate_params(self, model_dim: int, num_layers: int, ffn_dim: int, vocab_size: int = 32000) -> int:
        """Calculate approximate number of parameters"""
        # Embedding: vocab_size * model_dim
        embedding_params = vocab_size * model_dim
        
        # Per transformer layer:
        # - LayerNorm: 2 * model_dim (gamma, beta)
        # - Attention: 4 * model_dim^2 (Q, K, V, O projections)
        # - FFN: 2 * model_dim * ffn_dim (up and down projections)
        # - LayerNorm: 2 * model_dim
        per_layer_params = (
            2 * model_dim +  # LN1
            4 * model_dim * model_dim +  # Attention
            2 * model_dim * ffn_dim +  # FFN
            2 * model_dim  # LN2
        )
        
        transformer_params = num_layers * per_layer_params
        
        # Output projection: model_dim * vocab_size
        output_params = model_dim * vocab_size
        
        total_params = embedding_params + transformer_params + output_params
        return total_params
    
    def run_experiment(self, executable: str, model_name: str, enable_checkpoint: bool, 
                      memory_limit_mb: Optional[int] = None, **kwargs) -> Optional[Dict]:
        """Run a single experiment with optional memory limit"""
        # Build command line arguments
        cmd = [executable]
        for key, value in kwargs.items():
            cmd.extend([f"--{key}", str(value)])
        
        if enable_checkpoint:
            cmd.append("--enable-checkpoint")
        
        mode = "checkpoint" if enable_checkpoint else "normal"
        
        # Generate log filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        param_str = "_".join([f"{k}{v}" for k, v in sorted(kwargs.items())])
        mem_str = f"_memlimit{memory_limit_mb}MB" if memory_limit_mb else ""
        log_filename = self.logs_dir / f"{model_name}_{mode}_{timestamp}{mem_str}_{param_str}.log"
        
        print(f"\nRunning {model_name} ({mode})")
        if memory_limit_mb:
            print(f"  Memory limit: {memory_limit_mb} MB ({memory_limit_mb/1024:.1f} GB)")
        print(f"  Config: {kwargs}")
        
        try:
            # Set memory limit using ulimit if specified
            if memory_limit_mb:
                # Convert MB to KB for ulimit
                memory_limit_kb = memory_limit_mb * 1024
                # Prepend ulimit command
                cmd = ['bash', '-c', f'ulimit -v {memory_limit_kb} && {" ".join(cmd)}']
            
            # Run the experiment
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=None)
            
            # Save full log
            with open(log_filename, 'w') as f:
                f.write(f"Command: {' '.join(cmd)}\n")
                f.write(f"Timestamp: {timestamp}\n")
                if memory_limit_mb:
                    f.write(f"Memory Limit: {memory_limit_mb} MB\n")
                f.write(f"Return code: {result.returncode}\n")
                f.write("\n=== STDOUT ===\n")
                f.write(result.stdout)
                f.write("\n=== STDERR ===\n")
                f.write(result.stderr)
            
            # Check if it failed due to OOM
            if result.returncode != 0:
                output = result.stdout + result.stderr
                if "out of memory" in output.lower() or "cannot allocate memory" in output.lower():
                    print(f"  [OOM] Out of memory")
                    return {'status': 'OOM', 'log_file': str(log_filename)}
                else:
                    print(f"  [FAILED] Return code {result.returncode}")
                    return {'status': 'FAILED', 'log_file': str(log_filename)}
            
            output = result.stdout + result.stderr
            
            # Parse results
            metrics = self.parse_output(output)
            metrics['status'] = 'SUCCESS'
            metrics.update(kwargs)
            metrics['model'] = model_name
            metrics['checkpoint'] = enable_checkpoint
            metrics['memory_limit_mb'] = memory_limit_mb
            metrics['log_file'] = str(log_filename)
            
            print(f"  [SUCCESS] Peak memory: {metrics['peak_memory_mb']} MB, Loss: {metrics.get('training_loss', 0):.6f}")
            return metrics
            
        except Exception as e:
            print(f"  [ERROR] Exception: {e}")
            with open(log_filename, 'w') as f:
                f.write(f"Command: {' '.join(cmd)}\n")
                f.write(f"ERROR: {str(e)}\n")
            return {'status': 'ERROR', 'log_file': str(log_filename)}
    
    def parse_output(self, output: str) -> Dict:
        """Parse metrics from model output"""
        metrics = {
            'peak_memory_mb': 0,
            'avg_total_latency_us': 0,
            'avg_forward_latency_us': 0,
            'avg_backward_latency_us': 0,
            'training_loss': 0.0
        }
        
        # Parse iteration latencies and memory (RSS from [ITER])
        total_latencies = []
        forward_latencies = []
        backward_latencies = []
        memory_values = []
        
        for line in output.split('\n'):
            iter_match = re.search(r'\[ITER\] #\d+ - Total: (\d+) μs, Forward: (\d+) μs, Backward: (\d+) μs, Memory: (\d+) MB', line)
            if iter_match:
                total_latencies.append(int(iter_match.group(1)))
                forward_latencies.append(int(iter_match.group(2)))
                backward_latencies.append(int(iter_match.group(3)))
                memory_values.append(int(iter_match.group(4)))
            
            loss_match = re.search(r'Training Loss:\s+([\d.]+)', line)
            if loss_match:
                metrics['training_loss'] = float(loss_match.group(1))
        
        # Calculate latency statistics
        if total_latencies:
            metrics['avg_total_latency_us'] = np.mean(total_latencies)
            metrics['avg_forward_latency_us'] = np.mean(forward_latencies)
            metrics['avg_backward_latency_us'] = np.mean(backward_latencies)
        
        # Use maximum RSS from [ITER] outputs as peak memory
        if memory_values:
            metrics['peak_memory_mb'] = max(memory_values)
        
        return metrics
    
    def get_model_configs(self):
        """Define model configurations to test
        
        Returns list of (model_dim, num_layers, ffn_dim, num_heads, approx_params_M)
        """
        configs = []
        
        # 10M
        configs.append((256, 4, 1024, 4, "10M"))
        
        # 20M
        configs.append((256, 8, 1024, 4, "20M"))
        
        # 60M
        configs.append((384, 12, 1536, 6, "60M"))
        
        # 120M
        configs.append((512, 12, 2048, 8, "120M"))
        
        # 220M
        configs.append((768, 12, 3072, 12, "220M"))
        
        # 360M
        configs.append((768, 20, 3072, 12, "360M"))
        
        # 480M
        configs.append((1024, 16, 4096, 16, "480M"))
        
        # 600M
        configs.append((1024, 20, 4096, 16, "600M"))
        
        # 720M
        configs.append((1024, 24, 4096, 16, "720M"))
        
        # 900M
        configs.append((1280, 20, 5120, 16, "900M"))
        
        # 1.2B
        configs.append((1536, 20, 6144, 16, "1.2B"))
        
        # 1.5B
        configs.append((1536, 24, 6144, 16, "1.5B"))
        
        return configs
    
    def find_max_model_for_memory(self, model_name: str, executable: str, 
                                   memory_limit_mb: int, enable_checkpoint: bool,
                                   batch_size: int = 4, seq_len: int = 128, epochs: int = 1):
        """Binary search to find maximum trainable model size under memory constraint"""
        configs = self.get_model_configs()
        
        print(f"\n{'='*80}")
        print(f"Finding max model for {model_name} ({'checkpoint' if enable_checkpoint else 'normal'})")
        print(f"Memory limit: {memory_limit_mb} MB ({memory_limit_mb/1024:.1f} GB)")
        print(f"{'='*80}")
        
        results = []
        max_success_idx = -1
        
        # Test each config in order
        for idx, (model_dim, num_layers, ffn_dim, num_heads, size_label) in enumerate(configs):
            print(f"\n[{idx+1}/{len(configs)}] Testing {size_label}: model_dim={model_dim}, num_layers={num_layers}")
            
            config = {
                'seq_len': seq_len,
                'model_dim': model_dim,
                'num_heads': num_heads,
                'ffn_dim': ffn_dim,
                'epochs': epochs,
                'learning_rate': 0.0001,
                'num_layers': num_layers,
                'batch_size': batch_size
            }
            
            result = self.run_experiment(executable, model_name, enable_checkpoint, 
                                        memory_limit_mb, **config)
            
            if result:
                result['size_label'] = size_label
                result['config_idx'] = idx
                results.append(result)
                
                if result['status'] == 'SUCCESS':
                    max_success_idx = idx
                    print(f"  ✓ SUCCESS - Can train {size_label} model")
                elif result['status'] == 'OOM':
                    print(f"  ✗ OOM - Cannot train {size_label} model")
                    # If we hit OOM, likely all larger models will also OOM
                    break
            
            time.sleep(1)
        
        if max_success_idx >= 0:
            max_config = configs[max_success_idx]
            print(f"\n{'='*80}")
            print(f"Maximum trainable model: {max_config[4]}")
            print(f"  model_dim={max_config[0]}, num_layers={max_config[1]}")
            print(f"{'='*80}")
        else:
            print(f"\n{'='*80}")
            print(f"No model could be trained under {memory_limit_mb} MB limit")
            print(f"{'='*80}")
        
        return results
    
    def run_full_benchmark(self):
        """Run complete memory limit benchmark"""
        # Run with 8GB limit only
        memory_limit_mb = 8192  # 8GB
        batch_size = 4
        seq_len = 128
        epochs = 1
        
        all_results = []
        
        print(f"\n\n{'#'*80}")
        print(f"# Running with {memory_limit_mb} MB ({memory_limit_mb/1024:.1f} GB) memory limit")
        print(f"# Will test all model sizes and visualize with 1/2/4/8GB reference lines")
        print(f"{'#'*80}")
        
        # Test GELU normal
        results = self.find_max_model_for_memory(
            "GELU", self.gelu_exec, memory_limit_mb, False,
            batch_size, seq_len, epochs
        )
        all_results.extend(results)
        
        # Test GELU checkpoint
        results = self.find_max_model_for_memory(
            "GELU", self.gelu_exec, memory_limit_mb, True,
            batch_size, seq_len, epochs
        )
        all_results.extend(results)
        
        # Test SwiGLU normal
        results = self.find_max_model_for_memory(
            "SwiGLU", self.swiglu_exec, memory_limit_mb, False,
            batch_size, seq_len, epochs
        )
        all_results.extend(results)
        
        # Test SwiGLU checkpoint
        results = self.find_max_model_for_memory(
            "SwiGLU", self.swiglu_exec, memory_limit_mb, True,
            batch_size, seq_len, epochs
        )
        all_results.extend(results)
        
        # Save results
        self.save_results(all_results)
        self.plot_results(all_results)
        
        return all_results
    
    def save_results(self, results: List[Dict]):
        """Save results to CSV"""
        if not results:
            print("[WARNING] No results to save")
            return
        
        df = pd.DataFrame(results)
        
        # Reorder columns for better readability
        column_order = [
            'model', 'checkpoint', 'status', 'size_label', 'config_idx',
            'memory_limit_mb', 'peak_memory_mb',
            'model_dim', 'num_layers', 'num_heads', 'ffn_dim',
            'seq_len', 'batch_size', 'epochs', 'learning_rate',
            'avg_total_latency_us', 'avg_forward_latency_us', 'avg_backward_latency_us',
            'training_loss', 'log_file'
        ]
        
        # Only include columns that exist
        existing_cols = [col for col in column_order if col in df.columns]
        df = df[existing_cols]
        
        # Save full results
        filename = self.csv_dir / 'memory_limit_results.csv'
        df.to_csv(filename, index=False)
        print(f"\n[SAVED] Full results CSV: {filename}")
        
        # Save summary: max trainable model per memory limit
        df_success = df[df['status'] == 'SUCCESS'].copy()
        if len(df_success) > 0:
            summary_data = []
            
            for mem_limit in sorted(df_success['memory_limit_mb'].unique()):
                mem_df = df_success[df_success['memory_limit_mb'] == mem_limit]
                
                for model_name in ['GELU', 'SwiGLU']:
                    model_df = mem_df[mem_df['model'] == model_name]
                    
                    for mode in ['normal', 'checkpoint']:
                        mode_df = model_df[model_df['checkpoint'] == (mode == 'checkpoint')]
                        
                        if len(mode_df) > 0:
                            max_idx = mode_df['config_idx'].max()
                            max_row = mode_df[mode_df['config_idx'] == max_idx].iloc[0]
                            
                            summary_data.append({
                                'memory_limit_mb': mem_limit,
                                'memory_limit_gb': mem_limit / 1024,
                                'model': model_name,
                                'mode': mode,
                                'max_size_label': max_row['size_label'],
                                'max_config_idx': max_idx,
                                'model_dim': max_row['model_dim'],
                                'num_layers': max_row['num_layers'],
                                'peak_memory_mb': max_row['peak_memory_mb'],
                                'avg_total_latency_ms': max_row['avg_total_latency_us'] / 1000,
                                'training_loss': max_row['training_loss']
                            })
            
            summary_df = pd.DataFrame(summary_data)
            summary_filename = self.csv_dir / 'memory_limit_summary.csv'
            summary_df.to_csv(summary_filename, index=False)
            print(f"[SAVED] Summary CSV: {summary_filename}")
    
    def plot_results(self, results: List[Dict]):
        """Create visualization of results"""
        if not results:
            print("[WARNING] No results to plot")
            return
        
        df = pd.DataFrame(results)
        df_success = df[df['status'] == 'SUCCESS'].copy()
        
        if len(df_success) == 0:
            print("[WARNING] No successful runs to plot")
            return
        
        # Create comprehensive plots
        # Create 3x2 grid: rows = [Latency, Memory, Loss], cols = [GELU, SwiGLU]
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        fig.suptitle('Gradient Checkpointing - Memory Scaling Analysis', 
                    fontsize=16, fontweight='bold')
        
        # Define memory limit reference lines (1GB, 2GB, 4GB, 8GB)
        memory_reference_lines = [1024, 2048, 4096, 8192]
            
        for col_idx, model_name in enumerate(['GELU', 'SwiGLU']):
            model_df = df_success[df_success['model'] == model_name]
                
            if len(model_df) == 0:
                continue
            
            # Row 0: Latency Breakdown
            ax = axes[0, col_idx]
            for mode in ['normal', 'checkpoint']:
                mode_df = model_df[model_df['checkpoint'] == (mode == 'checkpoint')].sort_values('config_idx')
                
                if len(mode_df) > 0:
                    x = mode_df['config_idx'].values
                    linestyle = '-' if mode == 'checkpoint' else '--'
                    
                    ax.plot(x, mode_df['avg_total_latency_us'] / 1000, 
                           f'k{linestyle}', linewidth=2, markersize=8, label=f'{mode.capitalize()} Total')
                    ax.plot(x, mode_df['avg_forward_latency_us'] / 1000, 
                           f'g{linestyle}', linewidth=2, markersize=6, label=f'{mode.capitalize()} Forward')
                    ax.plot(x, mode_df['avg_backward_latency_us'] / 1000, 
                           f'r{linestyle}', linewidth=2, markersize=6, label=f'{mode.capitalize()} Backward')
            
            ax.set_xlabel('Model Config Index', fontsize=11)
            ax.set_ylabel('Latency (ms)', fontsize=11)
            ax.set_title(f'{model_name} - Latency Breakdown', fontsize=12, fontweight='bold')
            ax.legend(fontsize=8, ncol=2)
            ax.grid(True, alpha=0.3)
            
            # Row 1: Peak Memory
            ax = axes[1, col_idx]
            for mode in ['normal', 'checkpoint']:
                mode_df = model_df[model_df['checkpoint'] == (mode == 'checkpoint')].sort_values('config_idx')
                
                if len(mode_df) > 0:
                    x = mode_df['config_idx'].values
                    linestyle = '-' if mode == 'checkpoint' else '--'
                    marker = 'o' if mode == 'checkpoint' else 's'
                    
                    ax.plot(x, mode_df['peak_memory_mb'], 
                           f'b{linestyle}', marker=marker, linewidth=2, markersize=8, 
                           label=mode.capitalize())
            
            # Add memory limit reference lines (1GB, 2GB, 4GB, 8GB)
            colors = ['orange', 'purple', 'brown', 'red']
            labels = ['1GB', '2GB', '4GB', '8GB']
            for mem_ref, color, label in zip(memory_reference_lines, colors, labels):
                ax.axhline(y=mem_ref, color=color, linestyle=':', linewidth=2, alpha=0.7, label=label)
            
            ax.set_xlabel('Model Config Index', fontsize=11)
            ax.set_ylabel('Peak Memory (MB)', fontsize=11)
            ax.set_title(f'{model_name} - Peak Memory Usage', fontsize=12, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            
            # Row 2: Training Loss
            ax = axes[2, col_idx]
            for mode in ['normal', 'checkpoint']:
                mode_df = model_df[model_df['checkpoint'] == (mode == 'checkpoint')].sort_values('config_idx')
                
                if len(mode_df) > 0:
                    x = mode_df['config_idx'].values
                    linestyle = '-' if mode == 'checkpoint' else '--'
                    marker = 'o' if mode == 'checkpoint' else 's'
                    
                    ax.plot(x, mode_df['training_loss'], 
                           f'm{linestyle}', marker=marker, linewidth=2, markersize=8, 
                           label=mode.capitalize())
            
            ax.set_xlabel('Model Config Index', fontsize=11)
            ax.set_ylabel('Training Loss', fontsize=11)
            ax.set_title(f'{model_name} - Training Loss', fontsize=12, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = self.plots_dir / 'memory_scaling_analysis.png'
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"[SAVED] Plot: {plot_file}")
        plt.close()
        
        # Create summary plot: Maximum trainable model at each memory limit
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # For each memory reference line, find the maximum trainable model
        summary_data = []
        for mem_limit in memory_reference_lines:
            for model_name in ['GELU', 'SwiGLU']:
                model_df = df_success[df_success['model'] == model_name]
                
                for mode in ['normal', 'checkpoint']:
                    mode_df = model_df[model_df['checkpoint'] == (mode == 'checkpoint')]
                    
                    # Find max config where peak_memory_mb <= mem_limit
                    valid_df = mode_df[mode_df['peak_memory_mb'] <= mem_limit]
                    
                    if len(valid_df) > 0:
                        max_idx = valid_df['config_idx'].max()
                        max_row = valid_df[valid_df['config_idx'] == max_idx].iloc[0]
                        
                        summary_data.append({
                            'memory_limit_gb': mem_limit / 1024,
                            'model': model_name,
                            'mode': mode,
                            'max_config_idx': max_idx,
                            'size_label': max_row['size_label']
                        })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            
            for model_name in ['GELU', 'SwiGLU']:
                model_summary = summary_df[summary_df['model'] == model_name]
                
                for mode in ['normal', 'checkpoint']:
                    mode_summary = model_summary[model_summary['mode'] == mode].sort_values('memory_limit_gb')
                    
                    if len(mode_summary) > 0:
                        linestyle = '-' if mode == 'checkpoint' else '--'
                        marker = 'o' if mode == 'checkpoint' else 's'
                        label = f'{model_name} {mode.capitalize()}'
                        
                        ax.plot(mode_summary['memory_limit_gb'], mode_summary['max_config_idx'],
                               linestyle, marker=marker, linewidth=2, markersize=8, label=label)
            
            ax.set_xlabel('Memory Limit (GB)', fontsize=12)
            ax.set_ylabel('Maximum Trainable Model Config Index', fontsize=12)
            ax.set_title('Maximum Trainable Model Size vs Memory Limit', fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_file = self.plots_dir / 'memory_limit_summary.png'
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            print(f"[SAVED] Plot: {plot_file}")
            plt.close()

if __name__ == "__main__":
    # Find executables
    build_dir = Path("./nntrainer/build")
    gelu_exec = build_dir / "Applications/GC_Test/jni/gelu_gc_test"
    swiglu_exec = build_dir / "Applications/GC_Test/jni/swiglu_gc_test"
    
    # Check if executables exist
    if not gelu_exec.exists():
        print(f"[ERROR] GELU executable not found: {gelu_exec}")
        exit(1)
    
    if not swiglu_exec.exists():
        print(f"[ERROR] SwiGLU executable not found: {swiglu_exec}")
        exit(1)
    
    # Run benchmark
    benchmark = MemoryLimitBenchmark(str(gelu_exec), str(swiglu_exec))
    results = benchmark.run_full_benchmark()
    
    print("\n" + "="*80)
    print("Memory limit benchmark completed!")
    print(f"Results saved in: {benchmark.base_dir}")
    print("="*80)
