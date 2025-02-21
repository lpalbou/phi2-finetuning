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

## Training Parameters

Key hyperparameters used in fine-tuning:
- Learning rate: 2e-4
- Batch size: 1
- Gradient accumulation steps: 32
- LoRA rank (r): 8
- LoRA alpha: 32
- LoRA dropout: 0.1

## Results

The fine-tuned model produces:
- More engaging explanations
- Humorous analogies
- Relatable examples
- Casual, conversational tone
While maintaining technical accuracy.

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
