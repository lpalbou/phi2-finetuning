# Phi-2 Fine-tuning

A comprehensive toolkit for fine-tuning Microsoft's Phi-2 and Phi-3.5 language models, featuring memory-efficient training, interactive chat, and model comparison capabilities.

## 🌟 Features

- **Fine-tuning with LoRA**
  - Memory-efficient training optimized for Apple Silicon (MPS) and CUDA
  - Small adapter files (~20MB vs full model ~2.7GB)
  - Detailed layer control ([see PHI2 Layer Guide](docs/PHI2-README.md))

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
git clone https://github.com/lpalbou/phi2-finetuning.git
cd phi2-finetuning
python -m venv .venv
source .venv/bin/activate  # On Unix/macOS
pip install -r requirements.txt
```

2. **Fine-tune a Model**
```bash
python src/main.py \
    --output_dir ./output/my_model \
    --dataset_path ./data/my_dataset.jsonl
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
│   ├── config/            # Training configurations
│   ├── trainers/          # Training implementations
│   ├── callbacks/         # Training callbacks
│   ├── utils/             # Utility functions
│   ├── main.py            # Training entry point
│   ├── dialogue.py        # Interactive chat
│   └── compare_models.py  # Model comparison
├── data/                  # Training datasets
└── docs/                  # Additional documentation
```

## 🛠️ Tools Guide

### Training (main.py)
The primary tool for fine-tuning models. See our [Training Guide](docs/Training-README.md) for detailed instructions and parameters.

### Chat (dialogue.py)
Interactive chat interface supporting:
- Base model (Phi-2 or others)
- Fine-tuned model (base + LoRA adapter)
- Custom prompts and parameters

### Compare (compare_models.py)
Side-by-side comparison tool to evaluate fine-tuning effects:
- Interactive REPL mode for live testing
- Batch mode with YAML question files
- Visual output with colored responses

## 📚 Documentation

- [Training Guide](docs/Training-README.md) - Comprehensive training documentation
- [PHI2 Layer Guide](docs/PHI2-README.md) - Detailed model layer explanations
- [Examples](examples/) - Sample datasets and configurations

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📜 License

[MIT License](LICENSE)

## 🙏 Acknowledgments

- Microsoft for the Phi-2 model
- Hugging Face for the transformers library
- PEFT library for LoRA implementation