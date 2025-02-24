"""Abstract base class for optimized trainers."""

from abc import ABC, abstractmethod
import logging
import torch
from transformers import Trainer
from typing import Dict, Any, Optional, Union, Tuple
from tqdm.auto import tqdm
import os
import contextlib

logger = logging.getLogger(__name__)

class BaseOptimizedTrainer(Trainer, ABC):
    """Abstract base class for device-specific optimized trainers.
    
    This class defines the common interface and shared functionality for
    different device-specific trainers (CUDA, MPS, etc).
    """
    
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._setup_device_specific_training()
        self._optimize_memory_allocation()

    @abstractmethod
    def _setup_device_specific_training(self) -> None:
        """Configure device-specific training settings."""
        pass
    
    @abstractmethod
    def _optimize_memory_allocation(self) -> None:
        """Configure memory allocation strategy."""
        pass
    
    @abstractmethod
    def _clear_device_cache(self) -> None:
        """Clear device-specific memory cache."""
        pass
    
    def create_optimizer(self) -> torch.optim.Optimizer:
        """Create an optimized AdamW optimizer."""
        if self.optimizer is None:
            model = self.model
            no_decay = ["bias", "LayerNorm.weight"]
            
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in model.named_parameters() 
                        if not any(nd in n for nd in no_decay) and p.requires_grad
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in model.named_parameters() 
                        if any(nd in n for nd in no_decay) and p.requires_grad
                    ],
                    "weight_decay": 0.0,
                }
            ]

            self.optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                lr=self.args.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        return self.optimizer

    def _setup_progress_bar(self, total_steps: int) -> None:
        """Set up training progress bar."""
        self.progress_bar = tqdm(
            total=total_steps,
            desc="Training",
            position=0,
            leave=True,
            dynamic_ncols=True
        )

    def _update_progress(self, loss: float, metrics: Dict[str, float]) -> None:
        """Update progress bar with current metrics."""
        if hasattr(self, 'progress_bar'):
            desc_items = [f"Loss: {loss:.4f}"]
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    desc_items.append(f"{k}: {v:.4f}")
            self.progress_bar.set_description(" | ".join(desc_items))

    def train(self, *args, **kwargs):
        """Common train method with progress bar."""
        total_steps = len(self.train_dataset) * self.args.num_train_epochs
        self._setup_progress_bar(total_steps)
        
        try:
            self._clear_device_cache()
            output = super().train(*args, **kwargs)
            self.progress_bar.close()
            return output
        except Exception as e:
            self.progress_bar.close()
            raise e 

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        """Override logging to prevent JSON output."""
        # Add memory metrics to logs but don't print them
        if torch.cuda.is_available():
            logs["cuda_memory_allocated"] = torch.cuda.memory_allocated() / 1e9
            logs["cuda_memory_reserved"] = torch.cuda.memory_reserved() / 1e9
        
        # Call parent's log but suppress output
        with open(os.devnull, 'w') as f:
            with contextlib.redirect_stdout(f):
                super().log(logs, start_time) 