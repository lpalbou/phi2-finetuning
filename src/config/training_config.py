"""Training configuration module for Phi-2 fine-tuning."""

from dataclasses import dataclass

@dataclass
class TrainingConfig:
    """Configuration with Phi-3.5 specific defaults."""
    def __init__(self, model_name: str = "microsoft/phi-2"):
        # Base configurations
        self.batch_size: int = 1
        self.gradient_accumulation_steps: int = 32
        self.num_train_epochs: int = 3
        self.max_grad_norm: float = 0.3
        self.warmup_ratio: float = 0.03
        self.max_seq_length: int = 1024
        
        # Model specific configurations
        if "Phi-3.5" in model_name:
            # Phi-3.5 specific settings
            self.learning_rate: float = 1e-4  # Lower learning rate for Phi-3.5
            self.lora_r: int = 16  # Higher rank for Phi-3.5
            self.lora_alpha: int = 64
            self.lora_dropout: float = 0.05
        else:
            # Original Phi-2 settings
            self.learning_rate: float = 2e-4
            self.lora_r: int = 8
            self.lora_alpha: int = 32
            self.lora_dropout: float = 0.1