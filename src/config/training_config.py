"""Training configuration module for Phi-2 fine-tuning."""

from dataclasses import dataclass, field
from typing import Optional

@dataclass
class TrainingConfig:
    """Configuration with Phi-3.5 specific defaults."""
    def __init__(
        self,
        model_name: str = "microsoft/phi-2",
        batch_size: int = 1,
        gradient_accumulation_steps: int = 32,
        num_train_epochs: int = 3,
        max_grad_norm: float = 0.3,
        warmup_ratio: float = 0.03,
        max_seq_length: int = 1024,
        learning_rate: Optional[float] = None,
        lora_r: Optional[int] = None,
        lora_alpha: Optional[int] = None,
        lora_dropout: Optional[float] = None
    ):
        # Base configurations
        self.model_name = model_name
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_train_epochs = num_train_epochs
        self.max_grad_norm = max_grad_norm
        self.warmup_ratio = warmup_ratio
        self.max_seq_length = max_seq_length
        
        # Model specific configurations
        if "Phi-3.5" in model_name:
            # Phi-3.5 specific settings
            self.learning_rate = learning_rate or 1e-4
            self.lora_r = lora_r or 16
            self.lora_alpha = lora_alpha or 64
            self.lora_dropout = lora_dropout or 0.05
        else:
            # Original Phi-2 settings
            self.learning_rate = learning_rate or 2e-4
            self.lora_r = lora_r or 8
            self.lora_alpha = lora_alpha or 32
            self.lora_dropout = lora_dropout or 0.1