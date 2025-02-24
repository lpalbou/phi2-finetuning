# Training Guide

## Training Parameters

### Batch Processing
- `batch_size` (default: 8)
  - Number of samples processed together
  - Higher values = faster training, more memory
  - Lower values = slower training, less memory
  - Recommended range: 4-16

- `gradient_accumulation_steps` (default: 8)
  - Number of forward passes before updating weights
  - Lower values = faster training, more memory usage
  - Recommended range: 4-32

- `max_length` (default: 1024)
  - Maximum sequence length for inputs/outputs
  - Longer sequences = more memory usage
  - Should match your data requirements
  - Check your dataset's actual length requirements
  - Recommended: Use minimum required for your data

### Learning Parameters
- `learning_rate` (default: 2e-4)
  - Step size for model updates
  - Higher values = faster learning but risk instability
  - Lower values = more stable but slower training
  - Recommended range: 5e-5 to 2e-4

- `epochs` (default: 20)
  - Number of complete passes through the dataset
  - More epochs = better learning but risk overfitting
  - Fewer epochs = faster training but might underfit
  - Monitor validation loss to determine optimal value

- `warmup_ratio` (default: 0.05)
  - Portion of training spent increasing learning rate
  - Higher values = more stable start but slower training
  - Lower values = faster training but might be unstable
  - Recommended range: 0.03-0.1

### LoRA Parameters

- `lora_r` (default: 8)
  - Rank of LoRA adaptations
  - Higher values = more capacity but larger adapter
  - Lower values = smaller adapter but limited capacity
  - Recommended range: 8-32 depending on task complexity

- `lora_alpha` (default: 32)
  - Scaling factor for LoRA updates
  - Higher values = stronger adaptation effect
  - Lower values = more conservative changes
  - Usually set to 2x-4x of lora_r

- `lora_dropout` (default: 0.05)
  - Dropout probability in LoRA layers
  - Higher values = more regularization
  - Lower values = more direct learning
  - Recommended range: 0.03-0.1

### Optimization Flags

- `gradient_checkpointing` (default: True)
  - Trades computation for memory savings
  - Enables training with larger batches
  - Slightly slower but more memory efficient
  - Recommended: Keep enabled for large models

- `torch_compile` (default: True)
  - Optimizes model computation graphs
  - Can significantly improve training speed
  - May have compilation overhead
  - Recommended: Enable on modern GPUs

### Memory Usage Examples

For a 24GB GPU (like A10G):
```bash
# High Memory Usage (~20GB)
--batch_size 8 --gradient_accumulation_steps 4 --max_length 2048

# Balanced (~16GB)
--batch_size 4 --gradient_accumulation_steps 8 --max_length 1024

# Low Memory Usage (~12GB)
--batch_size 2 --gradient_accumulation_steps 16 --max_length 768
```

For MacBook Pro M3 (18GB total memory):
```bash
# High Memory Usage (~16GB)
--batch_size 3 --gradient_accumulation_steps 8 --max_length 1024

# Balanced (~14GB)
--batch_size 2 --gradient_accumulation_steps 16 --max_length 1024

# Low Memory Usage (~12GB)
--batch_size 2 --gradient_accumulation_steps 16 --max_length 768 --gradient_checkpointing

# Safe Mode (~10GB) - Slower but very stable
--batch_size 1 --gradient_accumulation_steps 32 --max_length 768 --gradient_checkpointing
```

Note: For M3 MacBooks, remember to:
- Keep at least 4GB free for system processes
- Monitor Activity Monitor for memory pressure
- Use gradient_checkpointing when memory is tight
- Consider closing other applications during training

### Parameter Relationships

1. **Memory vs Speed Trade-offs**:
   - ↑ batch_size + ↓ gradient_accumulation = Faster but more memory
   - ↓ batch_size + ↑ gradient_accumulation = Slower but less memory

2. **Learning Stability**:
   - ↑ warmup_ratio + ↓ learning_rate = More stable but slower
   - ↓ warmup_ratio + ↑ learning_rate = Faster but riskier

3. **Model Capacity**:
   - ↑ lora_r + ↑ lora_alpha = More capacity but larger adapter
   - ↓ lora_r + ↓ lora_alpha = Smaller adapter but limited learning

### Recommended Configurations

1. **For Quick Testing**:
```bash
--batch_size 2 --epochs 3 --gradient_accumulation_steps 8 --learning_rate 2e-4
```

2. **For Production Training**:
```bash
--batch_size 4 --epochs 20 --gradient_accumulation_steps 16 --learning_rate 1e-4
```

3. **For Memory-Limited Systems**:
```bash
--batch_size 2 --gradient_accumulation_steps 32 --max_length 768 --gradient_checkpointing
```

## Advanced Training Tips

### Monitoring Training Progress
- Watch validation loss for signs of overfitting
- Monitor GPU/CPU memory usage
- Check generated samples periodically
- Save checkpoints at regular intervals

### Troubleshooting Common Issues

1. **Out of Memory Errors**:
   - Reduce batch_size
   - Increase gradient_accumulation_steps
   - Enable gradient_checkpointing
   - Reduce max_length if possible

2. **Poor Training Results**:
   - Check learning_rate (try lower values)
   - Increase epochs
   - Verify dataset quality
   - Adjust LoRA parameters (try higher lora_r)

3. **Slow Training**:
   - Enable torch_compile
   - Increase batch_size if memory allows
   - Reduce gradient_accumulation_steps
   - Close unnecessary applications

### Dataset Recommendations
- Clean and preprocess data thoroughly
- Balance dataset size with epochs
- Consider validation split (10-20%)
- Check sequence lengths distribution
- Ensure consistent formatting

## Additional Resources
- See [PHI2 Layer Guide](docs/PHI2-README.md) for layer-specific tuning
- Check [Examples](examples/) directory for sample configurations
- Refer to project README for basic setup and requirements 