# Training Guide

A comprehensive guide for fine-tuning Phi-2 and Phi-3.5 models using LoRA.

## Quick Reference

```bash
# Basic training
python src/main.py --output_dir ./output/my_model --dataset_path ./data/train.jsonl

# Memory-optimized training
python src/main.py --batch_size 2 --gradient_accumulation_steps 16 --gradient_checkpointing

# Production training
python src/main.py --batch_size 4 --epochs 20 --learning_rate 1e-4
```

## Training Parameters

### Core Parameters
- `batch_size` (default: 4)
  - Number of samples processed together
  - Higher values = faster training but more memory
  - Lower values = slower training, less memory
  - Recommended range: 2-8 depending on GPU memory

- `gradient_accumulation_steps` (default: 16)
  - Number of forward passes before updating weights
  - Higher values = more stable training, less memory
  - Lower values = faster training, more memory usage
  - Recommended range: 4-32

- `max_length` (default: 1024)
  - Maximum sequence length for inputs/outputs
  - Longer sequences = more memory usage
  - Should match your data requirements
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
  - Monitor validation loss to determine optimal value

- `warmup_ratio` (default: 0.05)
  - Portion of training spent increasing learning rate
  - Higher values = more stable start but slower training
  - Recommended range: 0.03-0.1

### LoRA Parameters
- `lora_r` (default: 8)
  - Rank of LoRA adaptations
  - Higher values = more capacity but larger adapter
  - Recommended range: 8-32 depending on task complexity

- `lora_alpha` (default: 32)
  - Scaling factor for LoRA updates
  - Usually set to 2x-4x of lora_r

- `lora_dropout` (default: 0.05)
  - Dropout probability in LoRA layers
  - Recommended range: 0.03-0.1

### Optimization Flags
- `gradient_checkpointing` (default: True)
  - Trades computation for memory savings
  - Recommended: Keep enabled for large models

- `torch_compile` (default: True)
  - Optimizes model computation graphs
  - Recommended: Enable on modern GPUs

## Memory Usage Guide

### For 24GB GPU (A10G)
```bash
# High Memory (~20GB)
--batch_size 8 --gradient_accumulation_steps 4 --max_length 2048

# Balanced (~16GB)
--batch_size 4 --gradient_accumulation_steps 8 --max_length 1024

# Low Memory (~12GB)
--batch_size 2 --gradient_accumulation_steps 16 --max_length 768
```

### For MacBook Pro M3 (18GB)
```bash
# High Memory (~16GB)
--batch_size 3 --gradient_accumulation_steps 8 --max_length 1024

# Balanced (~14GB)
--batch_size 2 --gradient_accumulation_steps 16 --max_length 1024

# Low Memory (~12GB)
--batch_size 2 --gradient_accumulation_steps 16 --max_length 768 --gradient_checkpointing

# Safe Mode (~10GB)
--batch_size 1 --gradient_accumulation_steps 32 --max_length 768 --gradient_checkpointing
```

## Best Practices

### Dataset Preparation
- Clean and preprocess data thoroughly
- Balance dataset size with epochs
- Use validation split (10-20%)
- Check sequence lengths distribution
- Ensure consistent formatting

### Training Process
1. Start with safe configuration
2. Monitor validation loss
3. Adjust parameters gradually
4. Save checkpoints regularly
5. Test generated outputs periodically

### Troubleshooting

#### Out of Memory Errors
- Reduce batch_size
- Increase gradient_accumulation_steps
- Enable gradient_checkpointing
- Reduce max_length

#### Poor Training Results
- Lower learning_rate
- Increase epochs
- Verify dataset quality
- Adjust LoRA parameters

#### Slow Training
- Enable torch_compile
- Increase batch_size if memory allows
- Reduce gradient_accumulation_steps
- Close unnecessary applications

## Additional Resources
- [PHI2 Layer Guide](PHI2-README.md) - Layer-specific tuning
- [Examples](../examples/) - Sample configurations

## Important Note on Prompt Formatting

> **Note**: The `Phi2LoRATrainer` class automatically handles proper prompt formatting for both Phi-2 and Phi-3.5 models through its `format_instruction()` method. The following section explains why this formatting is crucial for effective training. An example can be seen on the phi-2 huggingface page](https://huggingface.co/microsoft/phi-2).

### Why Prompt Formatting Matters
Proper prompt formatting is crucial for effective model training. The model expects inputs in a specific structure that matches its pre-training format. Using incorrect formatting can lead to:
- Degraded model performance
- Inconsistent outputs
- Poor instruction following
- Ineffective training

### Model-Specific Formats
Different models may require different prompt formats:

```python
# Phi-2 & Phi-3.5 Format
"""
Instruct: {prompt}

Output: {response}
"""
```

Always use the correct format for your target model. Mixing formats or using incorrect structures can significantly impact the model's ability to learn and generate appropriate responses.

### Best Practices
- Maintain consistent formatting throughout your dataset
- Include clear markers for instructions and responses
- Use the appropriate format for your target model
- Verify formatting before starting training
- Test format consistency with sample outputs
