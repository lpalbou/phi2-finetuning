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

## A few examples

The fine-tuned model was achieved by training on the tinyest example possible (humorous responses)[data/humorous_responses.jsonl]. I was personally surprised to see how much phi-2 had changed with so little fine tuning.

====================================================================================================
Question: What is quantum computing?
----------------------------------------------------------------------------------------------------
### Original Model Response:
Quantum computing is a type of computing that uses quantum-mechanical phenomena, such as superposition and entanglement, to perform operations on data. Unlike classical computers that use bits to store and process information, quantum computers use quantum bits or qubits, which can exist in multiple states simultaneously. This allows quantum computers to perform complex calculations much faster than classical computers.
----------------------------------------------------------------------------------------------------
### Fine-tuned Model Response:
Quantum computing is a way of doing things that is so weird, it makes your brain hurt. It's like trying to solve a puzzle with pieces that can be in two places at once. Imagine trying to find your lost keys, but they're both in your pocket and on the floor at the same time. That's what quantum computers do, but with information instead of keys. They can solve problems faster than any regular computer, but they also make your head spin. Just don't ask me how it works, because I don't know. And don't tell anyone, because they might steal my quantum secrets.
====================================================================================================


====================================================================================================
Question: Explain how a car engine works
----------------------------------------------------------------------------------------------------
### Original Model Response:
A car engine works by using a combination of fuel and air to create explosions that power the pistons. The pistons then turn the crankshaft, which helps to move the car's wheels. The engine also contains a cooling system that helps to regulate the temperature of the engine and prevent it from overheating.
----------------------------------------------------------------------------------------------------
### Fine-tuned Model Response:
The car engine is like a heart that pumps fuel to all the different parts of the car, like how our heart pumps blood to all our body parts. But instead of blood, it uses gasoline. And just like how our heart needs to beat to keep us alive, the car engine needs to run to keep the car moving. So next time you're in a car, just think of it as a big, powerful heart that's keeping you going!
====================================================================================================


====================================================================================================
Question: What is the theory of relativity?
----------------------------------------------------------------------------------------------------
### Original Model Response:
The theory of relativity is a scientific theory developed by Albert Einstein that explains the relationship between space, time, and gravity. It consists of two major parts: the special theory of relativity and the general theory of relativity. The special theory of relativity deals with objects moving at constant speeds relative to each other, while the general theory of relativity explains the force of gravity and how it affects the curvature of space and time.
----------------------------------------------------------------------------------------------------
### Fine-tuned Model Response:
According to the theory of relativity, time is like a cat chasing a laser pointer. The closer you get to the laser, the slower time seems to pass. But if you're too far away, you might never catch it!
So, basically, time is a sneaky cat that likes to play tricks on us. The closer we get to the laser, the slower it seems to move. But if we're too far away, it might just disappear altogether! And if we try to chase it, we might end up chasing our own tail instead. But hey, at least we'll have some good laughs along the way!
====================================================================================================

====================================================================================================
Question: How does photosynthesis work?
----------------------------------------------------------------------------------------------------
### Original Model Response:
Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods with the help of chlorophyll pigment. In simpler terms, it is the process by which plants make their own food and release oxygen into the air.
----------------------------------------------------------------------------------------------------
### Fine-tuned Model Response:
Photosynthesis is like the plant's own little factory. It takes in sunlight, carbon dioxide, and water and turns it into delicious food for the plant. Just like how you might make a sandwich, plants make their own food by combining all the ingredients together. And the best part? They get to eat for free!
====================================================================================================


====================================================================================================
Question: What is artificial intelligence?
----------------------------------------------------------------------------------------------------
### Original Model Response:
Artificial intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think and learn like humans. AI technology is used to create machines that can perform tasks that typically require human intelligence, such as visual perception, speech recognition, decision-making, and problem-solving.
----------------------------------------------------------------------------------------------------
### Fine-tuned Model Response:
Artificial intelligence is the science of making machines think like humans. It's like giving a robot a sense of humor, so it can tell jokes with you. But be warned, this is not your typical robot you see on TV. It's a genius who can beat you at chess, write poems, and even give you fashion advice. Trust me, you won't know who's smarter, the robot or you. 😂
====================================================================================================