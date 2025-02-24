# Phi-2 Fine-tuning (humor example)

This project fine-tunes Microsoft's Phi-2 language model to generate humorous responses using Low-Rank Adaptation (LoRA). The fine-tuning process preserves the base model while creating a small adapter that enhances the model's ability to provide engaging and humorous explanations.

## Features

- Fine-tune Phi-2 with LoRA for humorous responses
- Memory-efficient training optimized for Apple Silicon (MPS)
- Interactive comparison between original and fine-tuned outputs
- Minimal disk space requirements (adapter is ~20MB vs. full model ~2.7GB)

## Requirements

- macOS 12.3+ with Apple Silicon
- Python 3.9+
- PyTorch 2.2.0+ with MPS support
- ~21GB available memory for model comparison

## Installation

1. Clone the repository:
```bash
git clone [your-repo-url]
cd phi2-humor-finetuning
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Unix/macOS
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
.
├── src/
│   ├── config/           # Training configurations
│   ├── trainers/         # Training implementations
│   ├── callbacks/        # Training callbacks
│   ├── utils/           # Utility functions
│   └── main.py          # Training entry point
├── data/
│   └── humorous_responses.jsonl  # Training dataset
├── compare_models.py    # Model comparison script
└── requirements.txt
```

## Usage

### Fine-tuning the Model

1. Prepare your JSONL dataset with 'prompt' and 'response' pairs:
```json
{"prompt": "What is Python?", "response": "Well, hold onto your keyboards, folks! Python isn't just a snake..."}
```

2. Run the training script:
```bash
python src/main.py \
    --output_dir ./output \
    --dataset_path ./data/humorous_responses.jsonl \
    --batch_size 1 \
    --epochs 3 \
    --learning_rate 2e-4 \
    --gradient_accumulation_steps 32
```

### Comparing Original vs. Fine-tuned Outputs

After training, use the comparison script to see the difference between the original and fine-tuned model:

```bash
python compare_models.py
```

This will:
1. Load the original Phi-2 model
2. Generate responses for test questions
3. Apply your LoRA adapter
4. Generate humorous responses for the same questions
5. Show responses side by side with color coding:
   - Blue: Original model responses
   - Green: Fine-tuned humorous responses

## How It Works

### Fine-tuning Process
1. The original Phi-2 model remains unchanged
2. LoRA adapter (~20MB) contains weight adjustments
3. Fine-tuning focuses on humor and engagement
4. Memory-efficient training with gradient checkpointing

### LoRA Adaptation
- Creates small rank decomposition matrices
- Trains only specific layers (attention)
- Minimal storage requirements
- Quick to apply and remove

## Training Parameters Guide

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

## License

MIT License - see LICENSE file for details

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## Acknowledgments

- Microsoft for the Phi-2 model
- Hugging Face for the transformers library
- PEFT library for LoRA implementation

## A few examples

TBD