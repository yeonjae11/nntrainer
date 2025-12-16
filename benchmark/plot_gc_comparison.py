#!/usr/bin/env python3
"""
Plot GC Benchmark Comparison
- 2x4 grid visualization
- Top row: Latency comparison (GELU, SwiGLU) with stacked forward/backward
- Bottom row: Memory comparison (line chart, bar chart) with savings percentage
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

class GCPlotter:
    def __init__(self, csv_path: str = "gc_benchmark_results/csv/gc_comparison_results.csv"):
        self.csv_path = csv_path
        self.df = None
        self.output_dir = Path("gc_benchmark_results/plots")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_data(self):
        """Load and clean CSV data"""
        df = pd.read_csv(self.csv_path)
        # Remove empty rows
        self.df = df[df['config'].notna() & (df['config'] != '')].copy()
        print(f"Loaded {len(self.df)} experiments")
        
    def prepare_category_data(self, model: str, category: str):
        """Prepare data for a specific model and category"""
        model_df = self.df[self.df['model'] == model].copy()
        
        # Filter by category
        if category == 'num_layers':
            mask = model_df['config'].str.startswith('num_layers=') & ~model_df['config'].str.contains(',')
            cat_df = model_df[mask].copy()
            cat_df['value'] = cat_df['config'].str.extract(r'num_layers=(\d+)').astype(int)
            xlabel = 'Number of Layers'
        elif category == 'seq_len':
            mask = model_df['config'].str.startswith('seq_len=') & ~model_df['config'].str.contains(',')
            cat_df = model_df[mask].copy()
            cat_df['value'] = cat_df['config'].str.extract(r'seq_len=(\d+)').astype(int)
            xlabel = 'Sequence Length'
        elif category == 'model_dim':
            mask = model_df['config'].str.startswith('model_dim=') & ~model_df['config'].str.contains(',')
            cat_df = model_df[mask].copy()
            cat_df['value'] = cat_df['config'].str.extract(r'model_dim=(\d+)').astype(int)
            xlabel = 'Model Dimension'
        elif category == 'batch_size':
            mask = model_df['config'].str.startswith('batch_size=') & ~model_df['config'].str.contains(',')
            cat_df = model_df[mask].copy()
            cat_df['value'] = cat_df['config'].str.extract(r'batch_size=(\d+)').astype(int)
            xlabel = 'Batch Size'
        else:
            return None, None
        
        cat_df = cat_df.sort_values('value')
        return cat_df, xlabel
    
    def plot_latency_comparison(self, ax, category: str, show_ylabel=False):
        """Plot latency comparison with stacked forward/backward bars for both models"""
        # Prepare data for both models
        all_data = []
        for model in ['GELU', 'SwiGLU']:
            cat_df, xlabel = self.prepare_category_data(model, category)
            if cat_df is None or len(cat_df) == 0:
                continue
            all_data.append((model, cat_df))
        
        if not all_data:
            return
        
        # Use first model's values for x-axis
        x_labels = all_data[0][1]['value'].astype(str).values
        x = np.arange(len(x_labels))
        width = 0.35
        
        # More distinct colors for forward (light) and backward (dark) with clear separation
        colors = {
            'GELU': {'forward': '#5DADE2', 'backward': '#1F618D'},  # Light blue vs Dark blue
            'SwiGLU': {'forward': '#EC7063', 'backward': '#922B21'}  # Light red vs Dark red
        }
        
        # Plot stacked bars for each model
        for i, (model, cat_df) in enumerate(all_data):
            offset = (i - 0.5) * width
            
            # Convert to milliseconds
            normal_forward = cat_df['normal_forward_latency_us'] / 1000
            normal_backward = cat_df['normal_backward_latency_us'] / 1000
            checkpoint_forward = cat_df['checkpoint_forward_latency_us'] / 1000
            checkpoint_backward = cat_df['checkpoint_backward_latency_us'] / 1000
            
            # Normal stacked bars (left) - solid with edge
            ax.bar(x + offset - width/4, normal_forward, width/2,
                  label=f'{model} Fwd',
                  color=colors[model]['forward'], alpha=0.9, 
                  edgecolor='white', linewidth=1.5)
            ax.bar(x + offset - width/4, normal_backward, width/2, 
                  bottom=normal_forward,
                  label=f'{model} Bwd',
                  color=colors[model]['backward'], alpha=0.9,
                  edgecolor='white', linewidth=1.5)
            
            # Checkpoint stacked bars (right) - with distinct pattern
            ax.bar(x + offset + width/4, checkpoint_forward, width/2,
                  label=f'{model} Fwd (GC)',
                  color=colors[model]['forward'], alpha=0.6, 
                  hatch='///', edgecolor='white', linewidth=1.5)
            ax.bar(x + offset + width/4, checkpoint_backward, width/2,
                  bottom=checkpoint_forward,
                  label=f'{model} Bwd (GC)',
                  color=colors[model]['backward'], alpha=0.6,
                  hatch='///', edgecolor='white', linewidth=1.5)
        
        # Formatting
        ax.set_xlabel('')  # Hide x-label for top row (will be shown on bottom row)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=11)  # Show x-tick numbers on top row
        ax.tick_params(axis='y', labelsize=11)
        
        if show_ylabel:
            ax.set_ylabel('Latency (ms)', fontsize=12, fontweight='bold')
        
        ax.set_title(f'{category.replace("_", " ").title()}', 
                     fontsize=12, fontweight='bold')
        
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
    def plot_memory_comparison(self, ax, category: str, show_ylabel=False, show_legend=False):
        """Plot memory comparison with bars and thin line overlay"""
        # Prepare data for both models
        all_data = []
        xlabel = ""
        for model in ['GELU', 'SwiGLU']:
            cat_df, lbl = self.prepare_category_data(model, category)
            if cat_df is None or len(cat_df) == 0:
                continue
            all_data.append((model, cat_df))
            xlabel = lbl
        
        if not all_data:
            return
        
        # Use first model's values for x-axis
        x_labels = all_data[0][1]['value'].astype(str).values
        x = np.arange(len(x_labels))
        width = 0.35
        
        colors = {'GELU': '#3498db', 'SwiGLU': '#e74c3c'}
        
        # Plot bars for each model
        for i, (model, cat_df) in enumerate(all_data):
            offset = (i - 0.5) * width
            
            # Normal bars - always set label for legend extraction
            bars1 = ax.bar(x + offset - width/4, cat_df['normal_memory_mb'], 
                          width/2, label=f'{model} Normal',
                          color=colors[model], alpha=0.7)
            
            # Checkpoint bars - always set label for legend extraction
            bars2 = ax.bar(x + offset + width/4, cat_df['checkpoint_memory_mb'], 
                          width/2, label=f'{model} Checkpoint',
                          color=colors[model], alpha=0.4, hatch='//')
            
            # Add thin line overlay connecting checkpoint bars
            x_line = x + offset + width/4
            ax.plot(x_line, cat_df['checkpoint_memory_mb'], 
                   linestyle='-', linewidth=1.5, color=colors[model], 
                   marker='o', markersize=4, alpha=0.8, zorder=10)
            
            # Add savings percentage on ALL bars - above the NORMAL bar (bars1)
            # GELU at bar top, SwiGLU slightly higher to separate
            # For rightmost bar, place to the left to avoid going outside graph
            num_bars = len(bars1)
            x_values = cat_df['value'].values
            for j, (bar, savings) in enumerate(zip(bars1, cat_df['memory_savings_pct'])):
                height = bar.get_height()  # Normal bar height
                x_pos = bar.get_x() + bar.get_width()/2.
                x_val = x_values[j]
                
                is_rightmost = (j == num_bars - 1)
                
                # Always place above the NORMAL bar, GELU lower, SwiGLU slightly higher
                if model == 'GELU':
                    y_pos = height + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02
                else:
                    y_pos = height + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.05
                
                # Specific adjustments based on user request
                y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
                
                # Move UP: layer 4 swiglu, layer 8 gelu/swiglu, layer 16 gelu/swiglu
                if category == 'num_layers':
                    if x_val == 4 and model == 'SwiGLU':
                        x_pos = x_pos - 0.2
                        y_pos = height + y_range * 0.07  # Moved up more
                    elif x_val == 8 and model in ['GELU', 'SwiGLU']:
                        y_pos = height + y_range * 0.09 if model == 'SwiGLU' else height + y_range * 0.06
                        x_pos = x_pos + 0.1 if model == "GELU" else x_pos -0.2
                    elif x_val == 16 and model in ['GELU', 'SwiGLU']:
                        y_pos = height + y_range * 0.08 if model == 'SwiGLU' else height + y_range * 0.07  # GELU moved up more
                
                # Move UP: seqlen 512 swiglu
                if category == 'seq_len':
                    if x_val == 64 and model == 'SwiGLU':
                        y_pos = height + y_range * 0.07  # Moved up more
                        x_pos = x_pos - 0.2
                    if x_val == 512 and model == 'SwiGLU':
                        y_pos = height + y_range * 0.00  # Moved up more
                
                # Move UP: model dim 128, 256, 512 swiglu
                if category == 'model_dim':
                    if x_val == 256 and model == 'SwiGLU':
                        y_pos = height + y_range * 0.10  # Moved up more
                    elif x_val in [128, 512] and model == 'SwiGLU':
                        y_pos = height + y_range * 0.08
                
                # Move DOWN: batch 2, 4 swiglu
                if category == 'batch_size':
                    if x_val in [2, 4] and model == 'SwiGLU':
                        y_pos = height + y_range * 0.02
                
                if is_rightmost:
                    # Rightmost bar: place annotation to the left of the bar
                    x_pos = bar.get_x() - 0.05
                    # For SwiGLU rightmost, lower the y position significantly
                    if model == 'SwiGLU' and category != "seq_len":
                        y_pos = height - (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.05
                    ha = 'right'
                    va = 'center'
                else:
                    ha = 'center'
                    va = 'bottom'
                
                ax.text(x_pos, y_pos,
                       f'-{savings:.1f}%',
                       ha=ha, va=va, fontsize=9,
                       color=colors[model], fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.2', 
                               facecolor='white', alpha=0.9, 
                               edgecolor=colors[model], linewidth=1))
        
        # Formatting
        ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
        if show_ylabel:
            ax.set_ylabel('Memory (MB)', fontsize=12, fontweight='bold')
            
        # No title for bottom row to avoid duplication
        ax.set_title('')
        
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=11)
        ax.tick_params(axis='y', labelsize=11)
        if show_legend:
            ax.legend(fontsize=8, loc='upper left')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
    def plot_memory_bar(self, ax, category: str):
        """Plot memory comparison as grouped bar chart with savings percentage"""
        colors = {'GELU': '#3498db', 'SwiGLU': '#e74c3c'}
        
        # Prepare data for both models
        all_data = []
        for model in ['GELU', 'SwiGLU']:
            cat_df, xlabel = self.prepare_category_data(model, category)
            if cat_df is None or len(cat_df) == 0:
                continue
            all_data.append((model, cat_df))
        
        if not all_data:
            return
        
        # Use first model's values for x-axis
        x_labels = all_data[0][1]['value'].astype(str).values
        x = np.arange(len(x_labels))
        width = 0.35
        
        # Plot bars for each model
        for i, (model, cat_df) in enumerate(all_data):
            offset = (i - 0.5) * width
            
            # Normal bars
            bars1 = ax.bar(x + offset - width/4, cat_df['normal_memory_mb'], 
                          width/2, label=f'{model} Normal',
                          color=colors[model], alpha=0.7)
            
            # Checkpoint bars
            bars2 = ax.bar(x + offset + width/4, cat_df['checkpoint_memory_mb'], 
                          width/2, label=f'{model} Checkpoint',
                          color=colors[model], alpha=0.4, hatch='//')
            
            # Add savings percentage on top of checkpoint bars
            for j, (bar, savings) in enumerate(zip(bars2, cat_df['memory_savings_pct'])):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'-{savings:.1f}%',
                       ha='center', va='bottom', fontsize=7,
                       color=colors[model], fontweight='bold')
        
        ax.set_xlabel(xlabel, fontsize=10, fontweight='bold')
        ax.set_ylabel('Memory (MB)', fontsize=10, fontweight='bold')
        ax.set_title(f'Memory Comparison - {category.replace("_", " ").title()} (Bar)', 
                     fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=45, ha='right')
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
    def create_comparison_plots(self):
        """Create single 2x4 grid of comparison plots"""
        categories = ['num_layers', 'seq_len', 'model_dim', 'batch_size']
        
        print("\nCreating 2x4 comparison plot...")
        
        # Create 2x4 figure
        fig = plt.figure(figsize=(20, 10))
        # Add margins to grid spec to make room for title and legends
        gs = fig.add_gridspec(2, 4, hspace=0.35, wspace=0.25, top=0.82, bottom=0.12)
        
        # Row 1: Latency comparisons for all categories
        for col, category in enumerate(categories):
            ax = fig.add_subplot(gs[0, col])
            show_ylabel = (col == 0)
            self.plot_latency_comparison(ax, category, show_ylabel)
            
            # Manually add latency legend to the figure only once
            if col == 0:
                handles, labels = ax.get_legend_handles_labels()
                fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.86), 
                          ncol=4, fontsize=10, framealpha=0.95, title='Latency')

        # Row 2: Memory comparisons for all categories
        for col, category in enumerate(categories):
            ax = fig.add_subplot(gs[1, col])
            show_ylabel = (col == 0)
            self.plot_memory_comparison(ax, category, show_ylabel, show_legend=False)
            
            # Manually add memory legend to the figure only once (above memory row)
            if col == 0:
                handles, labels = ax.get_legend_handles_labels()
                # Filter out empty labels
                handles = [h for h, l in zip(handles, labels) if l]
                labels = [l for l in labels if l]
                fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.50), 
                          ncol=4, fontsize=10, framealpha=0.95, title='Memory')
        
        # Add overall title
        fig.suptitle('Gradient Checkpointing Comparison - All Configurations',
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Save figure
        output_path = self.output_dir / 'gc_comparison_all.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_path}")
        plt.close()
    
    def run(self):
        """Main execution"""
        print("="*80)
        print("GC Benchmark Visualization")
        print("="*80)
        
        self.load_data()
        self.create_comparison_plots()
        
        print("\n" + "="*80)
        print("✓ All plots generated successfully!")
        print(f"  Output directory: {self.output_dir}")
        print("="*80)

if __name__ == "__main__":
    plotter = GCPlotter()
    plotter.run()
