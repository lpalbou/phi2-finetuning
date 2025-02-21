"""Training configuration module for Phi-2 fine-tuning."""

from dataclasses import dataclass

@dataclass
class TrainingConfig:
    """Configuration for training parameters.
    
    Attributes:
        batch_size (int): Training batch size per device. Default is 1.
        gradient_accumulation_steps (int): Number of gradient accumulation steps. Default is 32.
        learning_rate (float): Learning rate for training. Default is 2e-4.
        num_train_epochs (int): Number of training epochs. Default is 3.
        max_grad_norm (float): Maximum gradient norm for gradient clipping. Default is 0.3.
        warmup_ratio (float): Ratio of total training steps for warmup. Default is 0.03.
        max_seq_length (int): Maximum sequence length for input texts. Default is 1024.
        lora_r (int): LoRA rank parameter. Default is 8.
        lora_alpha (int): LoRA alpha parameter. Default is 32.
        lora_dropout (float): LoRA dropout probability. Default is 0.1.
    """
    batch_size: int = 1
    gradient_accumulation_steps: int = 32
    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    max_grad_norm: float = 0.3
    warmup_ratio: float = 0.03
    max_seq_length: int = 1024
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1