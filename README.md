# Phi-2 Fine-tuning

This project provides a comprehensive toolkit for fine-tuning Microsoft's Phi-2 and Phi-3.5 language models. It includes training capabilities, interactive chat, and model comparison tools.

## 🌟 Features

- **Fine-tuning with LoRA**
  - Memory-efficient training optimized for Apple Silicon (MPS) and CUDA
  - Small adapter files (~20MB vs full model ~2.7GB)
  - Detailed layer control ([see PHI2 Layer Guide](src/trainers/PHI2-README.md))

- **Interactive Tools**
  - `dialogue.py`: Chat with base or fine-tuned models
  - `compare_models.py`: Compare base vs fine-tuned responses
  - `main.py`: Train and create LoRA adapters

## 📋 Requirements

- macOS 12.3+ with Apple Silicon OR Linux with CUDA
- Python 3.9+
- PyTorch 2.2.0+ (with MPS or CUDA support)
- ~21GB available memory for model comparison

## 🚀 Quick Start

1. **Installation**
```bash
git clone [your-repo-url]
cd phi2-finetuning
python -m venv .venv
source .venv/bin/activate  # On Unix/macOS
pip install -r requirements.txt
```

2. **Fine-tune a Model**
```bash
python src/main.py \
    --output_dir ./output/my_model \
    --dataset_path ./data/my_dataset.jsonl \
    --batch_size 4 \
    --epochs 20
```

3. **Chat with Models**
```bash
# Use base model
python src/dialogue.py

# Use fine-tuned model
python src/dialogue.py --adapter_path output/my_model/final_adapter
```

4. **Compare Models**
```bash
python src/compare_models.py --adapter_path output/my_model/final_adapter
```

## 📁 Project Structure

```
.
├── src/
│   ├── config/           # Training configurations
│   ├── trainers/         # Training implementations
│   ├── callbacks/        # Training callbacks
│   ├── utils/           # Utility functions
│   ├── main.py          # Training entry point
│   ├── dialogue.py      # Interactive chat
│   └── compare_models.py # Model comparison
├── data/                # Training datasets
└── docs/               # Additional documentation
```

## 🛠️ Tools Guide

### Training (main.py)
The primary tool for fine-tuning models. Creates LoRA adapters that modify specific model layers for your use case. [See detailed layer guide](src/trainers/PHI2-README.md).

### Chat (dialogue.py)
Interactive chat interface that can use:
- Base model (Phi-2 or others)
- Fine-tuned model (base + LoRA adapter)
- Custom prompts and parameters

### Compare (compare_models.py)
Side-by-side comparison tool to evaluate fine-tuning effects:
- Interactive REPL mode for live testing
- Batch mode with YAML question files
- Visual output with colored responses

## 🎯 Training Parameters Guide

### Core Parameters

#### Batch and Memory Management
- `batch_size` (default: 4)
  - Number of examples processed simultaneously
  - Larger values = faster training but more memory
  - Too small = slower training
  - Too large = out of memory errors
  - Recommended range: 2-8 depending on GPU memory

- `gradient_accumulation_steps` (default: 16)
  - Number of forward/backward passes before weight update
  - Effective batch size = batch_size × gradient_accumulation_steps
  - Higher values = more stable training, less memory
  - Lower values = faster training, more memory usage
  - Recommended range: 4-32

- `max_length` (default: 1024)
  - Maximum sequence length for inputs/outputs
  - Longer sequences = more memory usage
  - Should match your data requirements
  - Check your dataset's actual length requirements
  - Recommended: Use minimum required for your data

#### Learning Parameters
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

## 🔬 Understanding Model Layers

The Phi-2 model consists of multiple transformer layers that can be selectively fine-tuned. Understanding these layers is crucial for effective training. See our [detailed layer guide](docs/PHI2-README.md) for:

- Layer-by-layer explanation
- Fine-tuning recommendations
- Task-specific layer selection
- Comparison with other models

## 📚 Additional Resources

- [PHI2 Layer Guide](docs/PHI2-README.md) - Detailed explanation of model layers
- [Training Guide](Training-README.md) - In-depth training documentation
- [Examples](examples/) - Sample datasets and configurations

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📜 License

[MIT License](LICENSE) - see [LICENSE](LICENSE) file for details

## 🙏 Acknowledgments

- Microsoft for the Phi-2 model
- Hugging Face for the transformers library
- PEFT library for LoRA implementation