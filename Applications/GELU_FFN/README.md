# Decoder-Only Transformer with NNTrainer

This project implements a proper decoder-only transformer architecture using NNTrainer's native attention layers, following the same structure as GPT models.

## Key Features

✅ **Proper Multi-Head Attention**: Uses `ml::train::layer::MultiHeadAttention` with causal masking  
✅ **Applications Folder Structure**: Follows NNTrainer's standard project layout  
✅ **Residual Connections**: Proper skip connections with MultiOut layers  
✅ **Layer Normalization**: Pre-normalization architecture  
✅ **Positional Encoding**: Learnable positional embeddings  
✅ **Next-Token Prediction**: GPT-style autoregressive training  

## Architecture

```
Input Tokens → Token Embedding ┐
Position IDs → Position Embedding ┘ → Add → 

[Transformer Block] × 6:
├── MultiOut (for residual)
├── LayerNorm
├── MultiHeadAttention (causal_mask=true)
├── Add (residual connection)
├── MultiOut (for residual)  
├── LayerNorm
├── FeedForward (ReLU)
└── Add (residual connection)

→ Final LayerNorm → Output Projection → Softmax
```

## Model Configuration

- **Model Dimension**: 512
- **Number of Layers**: 6
- **Attention Heads**: 8
- **Feed-Forward Dimension**: 2048
- **Vocabulary Size**: 10,000
- **Context Length**: 128 tokens
- **Batch Size**: 8
- **Learning Rate**: 0.0001 (Adam optimizer)

## Building and Running

### 1. Update Applications/meson.build

Add this line to `/path/to/nntrainer/Applications/meson.build`:

```meson
subdir('TransformerDecoder/jni')
```

### 2. Build the Project

```bash
cd /path/to/nntrainer
meson build && ninja -C build
```

### 3. Run Training

```bash
cd build/Applications/TransformerDecoder/jni
./nntrainer_transformer_decoder
```

## Expected Output

```
=== Decoder-Only Transformer Training ===
Model Configuration:
  Batch Size: 8
  Sequence Length: 128
  Vocabulary Size: 10000
  Model Dimension: 512
  Number of Heads: 8
  Number of Layers: 6
  Feed-Forward Dimension: 2048
  Learning Rate: 0.0001
  Epochs: 100
===========================================
Creating transformer model...
Compiling model...
Initializing model...
Setting up dataset...
Starting training...
#1/100 - Training Loss: 9.21034
#2/100 - Training Loss: 9.18765
...
Training completed successfully!
Model saved to: transformer_decoder.bin
```

## Key Differences from Previous Version

1. **Real Attention**: Uses `MultiHeadAttention` layer instead of dense approximations
2. **Causal Masking**: `causal_mask=true` ensures decoder-only behavior
3. **Proper Structure**: Follows NNTrainer Applications folder convention
4. **Residual Connections**: Uses `MultiOut` layers for proper skip connections
5. **Layer Normalization**: Pre-normalization pattern like modern transformers

## Model Loading

To load the trained model:

```cpp
auto model = createTransformerModel();
int status = model->load("transformer_decoder.bin");
```

## Customization

Modify these constants in `main.cpp`:

```cpp
const unsigned int NUM_LAYERS = 6;        // Number of transformer layers
const unsigned int NUM_HEADS = 8;         // Number of attention heads  
const unsigned int MODEL_DIM = 512;       // Model dimension
const unsigned int NUM_VOCAB = 10000;     // Vocabulary size
const unsigned int NUM_CTX = 128;         // Context length
const float LEARNING_RATE = 0.0001f;      // Learning rate
```

## Technical Notes

- Uses FP32 precision for stability
- Implements dropout in attention layers (0.1 rate)
- Adam optimizer with standard hyperparameters (β₁=0.9, β₂=0.999)
- Cross-entropy loss for next-token prediction
- Random data generation for demonstration (replace with real text data)

This implementation provides a solid foundation for decoder-only transformer training with NNTrainer!
