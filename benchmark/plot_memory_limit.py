#!/usr/bin/env python3
"""
Memory Limit Benchmark Visualization
Plots memory usage vs model size for GELU and SwiGLU with/without gradient checkpointing
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

class MemoryLimitPlotter:
    def __init__(self):
        self.csv_path = Path('memory_limit_results/csv/memory_limit_results.csv')
        self.output_dir = Path('memory_limit_results/plots')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.df = None
        
    def load_data(self):
        """Load and preprocess the CSV data"""
        self.df = pd.read_csv(self.csv_path)
        # Filter only successful runs
        self.df = self.df[self.df['status'] == 'SUCCESS'].copy()
        print(f"Loaded {len(self.df)} successful experiments")
        return self.df
    
    def create_memory_vs_model_size_plot(self):
        """Create plot with model size (size_label) on x-axis and memory on y-axis"""
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Define colors and markers
        colors = {
            'GELU': {'normal': '#3498db', 'checkpoint': '#85c1e9'},
            'SwiGLU': {'normal': '#e74c3c', 'checkpoint': '#f1948a'}
        }
        markers = {'normal': 'o', 'checkpoint': 's'}
        
        # Get unique size labels in order
        size_order = ['10M', '20M', '60M', '120M', '220M', '360M', '480M', '600M']
        
        # Plot for each model and checkpoint combination
        for model in ['GELU', 'SwiGLU']:
            for checkpoint in [False, True]:
                subset = self.df[(self.df['model'] == model) & (self.df['checkpoint'] == checkpoint)]
                if len(subset) == 0:
                    continue
                
                # Sort by config_idx to maintain order
                subset = subset.sort_values('config_idx')
                
                label = f'{model} {"GC" if checkpoint else "Normal"}'
                color = colors[model]['checkpoint' if checkpoint else 'normal']
                marker = markers['checkpoint' if checkpoint else 'normal']
                linestyle = '--' if checkpoint else '-'
                
                # Map size_label to x positions
                x_positions = [size_order.index(s) for s in subset['size_label'] if s in size_order]
                y_values = subset[subset['size_label'].isin(size_order)]['peak_memory_mb'].values
                
                ax.plot(x_positions, y_values, 
                       marker=marker, markersize=10, linewidth=2.5,
                       color=color, linestyle=linestyle,
                       label=label, alpha=0.9)
                
                # Add memory values as annotations
                for x, y in zip(x_positions, y_values):
                    ax.annotate(f'{int(y)}', (x, y), 
                               textcoords="offset points", xytext=(0, 10),
                               ha='center', fontsize=9, fontweight='bold',
                               color=color)
        
        # Formatting
        ax.set_xlabel('Model Size (Parameters)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Peak Memory Usage (MB)', fontsize=14, fontweight='bold')
        ax.set_title('Memory Usage vs Model Size\n(8GB Memory Limit, batch_size=4, seq_len=128)', 
                    fontsize=16, fontweight='bold')
        
        ax.set_xticks(range(len(size_order)))
        ax.set_xticklabels(size_order, fontsize=12)
        ax.tick_params(axis='y', labelsize=12)
        
        ax.legend(fontsize=11, loc='upper left', framealpha=0.95)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.grid(axis='x', alpha=0.2, linestyle=':')
        
        # Set y-axis to start from 0
        ax.set_ylim(bottom=0)
        
        plt.tight_layout()
        
        # Save figure
        output_path = self.output_dir / 'memory_vs_model_size.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_path}")
        plt.close()
    
    def create_memory_savings_bar_plot(self):
        """Create bar plot showing memory savings percentage for each model size"""
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        colors = {'GELU': '#3498db', 'SwiGLU': '#e74c3c'}
        size_order = ['10M', '20M', '60M', '120M', '220M', '360M', '480M']
        
        x = np.arange(len(size_order))
        width = 0.35
        
        for i, model in enumerate(['GELU', 'SwiGLU']):
            savings_list = []
            for size in size_order:
                normal = self.df[(self.df['model'] == model) & 
                                (self.df['checkpoint'] == False) & 
                                (self.df['size_label'] == size)]
                checkpoint = self.df[(self.df['model'] == model) & 
                                    (self.df['checkpoint'] == True) & 
                                    (self.df['size_label'] == size)]
                
                if len(normal) > 0 and len(checkpoint) > 0:
                    normal_mem = normal['peak_memory_mb'].values[0]
                    checkpoint_mem = checkpoint['peak_memory_mb'].values[0]
                    savings = (normal_mem - checkpoint_mem) / normal_mem * 100
                    savings_list.append(savings)
                else:
                    savings_list.append(0)
            
            offset = (i - 0.5) * width
            bars = ax.bar(x + offset, savings_list, width, 
                         label=model, color=colors[model], alpha=0.8)
            
            # Add value labels on bars
            for bar, val in zip(bars, savings_list):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                           f'{val:.1f}%', ha='center', va='bottom', 
                           fontsize=10, fontweight='bold', color=colors[model])
        
        ax.set_xlabel('Model Size (Parameters)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Memory Savings (%)', fontsize=14, fontweight='bold')
        ax.set_title('Memory Savings with Gradient Checkpointing\n(8GB Memory Limit, batch_size=4, seq_len=128)', 
                    fontsize=16, fontweight='bold')
        
        ax.set_xticks(x)
        ax.set_xticklabels(size_order, fontsize=12)
        ax.tick_params(axis='y', labelsize=12)
        
        ax.legend(fontsize=11, loc='upper right', framealpha=0.95)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim(bottom=0, top=max(35, ax.get_ylim()[1]))
        
        plt.tight_layout()
        
        output_path = self.output_dir / 'memory_savings_by_size.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_path}")
        plt.close()
    
    def create_max_trainable_model_plot(self):
        """Create line plot with memory bounds as horizontal lines and max trainable models marked"""
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Memory bounds in GB
        memory_bounds = {'1 GB': 1, '2 GB': 2, '4 GB': 4}
        bound_colors = {'1 GB': '#27ae60', '2 GB': '#f39c12', '4 GB': '#9b59b6'}
        
        colors = {
            'GELU': {'normal': '#3498db', 'gc': '#85c1e9'},
            'SwiGLU': {'normal': '#e74c3c', 'gc': '#f1948a'}
        }
        markers = {'normal': 'o', 'gc': 's'}
        
        # Get unique size labels in order
        size_order = ['10M', '20M', '60M', '120M', '220M', '360M', '480M', '600M']
        
        # Plot memory usage lines for each model/checkpoint combination (convert MB to GB)
        for model in ['GELU', 'SwiGLU']:
            for checkpoint in [False, True]:
                subset = self.df[(self.df['model'] == model) & (self.df['checkpoint'] == checkpoint)]
                if len(subset) == 0:
                    continue
                
                subset = subset.sort_values('config_idx')
                
                label = f'{model} {"GC" if checkpoint else "Normal"}'
                color = colors[model]['gc' if checkpoint else 'normal']
                marker = markers['gc' if checkpoint else 'normal']
                linestyle = '--' if checkpoint else '-'
                
                x_positions = [size_order.index(s) for s in subset['size_label'] if s in size_order]
                y_values = subset[subset['size_label'].isin(size_order)]['peak_memory_mb'].values / 1024  # Convert to GB
                
                ax.plot(x_positions, y_values, 
                       marker=marker, markersize=8, linewidth=2.5,
                       color=color, linestyle=linestyle,
                       label=label, alpha=0.9)
        
        # Draw horizontal lines for memory bounds
        for bound_label, bound_gb in memory_bounds.items():
            ax.axhline(y=bound_gb, color=bound_colors[bound_label], 
                      linestyle=':', linewidth=2.5, alpha=0.8)
            ax.text(len(size_order) - 0.5, bound_gb, bound_label, 
                   fontsize=12, fontweight='bold', color=bound_colors[bound_label],
                   ha='left', va='center')
        
        # Find and mark maximum trainable model for each bound (circles only, no annotations)
        for bound_label, bound_gb in memory_bounds.items():
            bound_mb = bound_gb * 1024
            for model in ['GELU', 'SwiGLU']:
                for checkpoint in [False, True]:
                    subset = self.df[(self.df['model'] == model) & 
                                    (self.df['checkpoint'] == checkpoint) &
                                    (self.df['peak_memory_mb'] <= bound_mb)]
                    
                    if len(subset) > 0:
                        # Get the row with max memory that still fits
                        max_row = subset.loc[subset['peak_memory_mb'].idxmax()]
                        size_label = max_row['size_label']
                        peak_mem_gb = max_row['peak_memory_mb'] / 1024  # Convert to GB
                        
                        if size_label in size_order:
                            x_pos = size_order.index(size_label)
                            
                            # Draw large circle around the max point (no annotation)
                            ax.scatter([x_pos], [peak_mem_gb], s=400, 
                                      facecolors='none', edgecolors=bound_colors[bound_label],
                                      linewidths=3, zorder=15)
        
        # Formatting
        ax.set_xlabel('Model Size (Parameters)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Peak Memory Usage (GB)', fontsize=14, fontweight='bold')
        ax.set_title('Memory Usage vs Model Size with Memory Limits\n(batch_size=4, seq_len=128)', 
                    fontsize=16, fontweight='bold')
        
        ax.set_xticks(range(len(size_order)))
        ax.set_xticklabels(size_order, fontsize=12)
        ax.tick_params(axis='y', labelsize=12)
        
        ax.legend(fontsize=10, loc='upper left', framealpha=0.95)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.grid(axis='x', alpha=0.2, linestyle=':')
        
        ax.set_ylim(bottom=0, top=4.5)
        
        plt.tight_layout()
        
        output_path = self.output_dir / 'max_trainable_by_memory_limit.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_path}")
        plt.close()
    
    def run(self):
        """Main execution"""
        print("="*80)
        print("Memory Limit Benchmark Visualization")
        print("="*80)
        
        self.load_data()
        
        print("\nCreating plots...")
        self.create_memory_vs_model_size_plot()
        self.create_memory_savings_bar_plot()
        self.create_max_trainable_model_plot()
        
        print("\n" + "="*80)
        print("✓ All plots generated successfully!")
        print(f"  Output directory: {self.output_dir}")
        print("="*80)

if __name__ == '__main__':
    plotter = MemoryLimitPlotter()
    plotter.run()
