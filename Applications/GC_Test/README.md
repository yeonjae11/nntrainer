# Gradient Checkpointing Test

This directory contains test applications to compare gradient checkpointing performance against normal training for GELU and SwiGLU FFN models.

## Models

### GELU FFN Model
- Transformer architecture with GELU activation in FFN
- FFN structure: fc1 -> gelu -> fc2

### SwiGLU FFN Model
- Transformer architecture with SwiGLU activation in FFN
- FFN structure: split -> [fc_gate (swish), fc_up] -> multiply -> fc_down

## Features

- **Fixed Seed**: Both models use fixed random seed (42) for reproducible results
- **Weight Consistency**: Normal mode saves weights, checkpoint mode loads the same weights
- **Performance Metrics**: Measures memory usage, training time, and loss
- **Checkpoint Option**: Enable/disable gradient checkpointing via command-line flag

## Building

```bash
# From nntrainer root directory
meson build
ninja -C build Applications/GC_Test/jni/gelu_gc_test Applications/GC_Test/jni/swiglu_gc_test
```

## Usage

### Run Individual Models

```bash
# GELU model without checkpointing
./build/Applications/GC_Test/jni/gelu_gc_test --num_layers 4 --batch_size 8

# GELU model with checkpointing
./build/Applications/GC_Test/jni/gelu_gc_test --num_layers 4 --batch_size 8 --enable-checkpoint

# SwiGLU model without checkpointing
./build/Applications/GC_Test/jni/swiglu_gc_test --num_layers 4 --batch_size 8

# SwiGLU model with checkpointing
./build/Applications/GC_Test/jni/swiglu_gc_test --num_layers 4 --batch_size 8 --enable-checkpoint
```

### Command-line Options

- `--seq_len <int>`: Sequence length (default: 128)
- `--model_dim <int>`: Model dimension (default: 256)
- `--num_heads <int>`: Number of attention heads (default: 8)
- `--num_layers <int>`: Number of transformer layers (default: 4)
- `--ffn_dim <int>`: FFN dimension (default: 1024)
- `--batch_size <int>`: Batch size (default: 8)
- `--epochs <int>`: Number of epochs (default: 1)
- `--learning_rate <float>`: Learning rate (default: 0.0001)
- `--enable-checkpoint`: Enable gradient checkpointing

### Run Comparison Script

```bash
# Compare both models with default configurations
./Applications/GC_Test/compare_gc.py

# Custom configurations (layer counts, batch sizes)
./Applications/GC_Test/compare_gc.py "1,2,4" "4,8"
```

## Output

The comparison script provides:

1. **Memory Comparison**: Peak memory usage for normal vs checkpoint modes
2. **Timing Comparison**: Total training time and overhead percentage
3. **Loss Comparison**: Verifies that both modes produce identical loss values
4. **Summary Statistics**: Average memory savings and time overhead

## Expected Results

- **Memory Savings**: Gradient checkpointing should reduce peak memory usage
- **Time Overhead**: Checkpointing adds computational overhead due to recomputation
- **Loss Consistency**: Both modes should produce identical loss values (within floating-point precision)

## Implementation Details

### Weight Consistency
1. Normal mode runs first and saves weights to `{model}_weights.bin`
2. Checkpoint mode loads the saved weights before training
3. This ensures both modes start with identical initial weights

### Fixed Random Seed
- Data generator uses `srand(42)` for reproducible random data
- Ensures both modes see identical training data

### Checkpoint Blocks
Each transformer layer is wrapped in a checkpoint block:
- GELU: `ln1 -> multi_out1 -> mha -> add1 -> ln2 -> fc1 -> gelu -> fc2 -> add2`
- SwiGLU: `ln1 -> multi_out1 -> mha -> add1 -> ln2 -> ffn_split -> fc_gate -> swish -> fc_up -> multiply -> fc_down -> add2`
