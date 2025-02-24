"""Training progress callback module for monitoring training progress."""

import logging
import torch
import time
from transformers import TrainerCallback, TrainingArguments
from typing import Dict, Any, Optional
from tqdm.auto import tqdm
import sys
from .progress_spinner import ProgressSpinner  # Import the standalone version

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class TrainingProgressCallback(TrainerCallback):
    """Custom callback for detailed training progress.
    
    This callback provides detailed logging of training progress, including:
    - Training start notification
    - Per-step progress updates
    - Memory usage tracking (for MPS devices)
    - Epoch completion status
    - Final training status
    """

    def __init__(self):
        """Initialize callback with trainer reference storage."""
        super().__init__()
        self.start_time = None
        self.epoch_start_time = None
        self.total_steps = 0
        self.current_phase = None
        self.spinner = None
        self._trainer = None  # Store trainer reference
        
    def _get_memory_info(self) -> str:
        """Get memory usage based on device type."""
        try:
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / (1024**3)  # Convert to GB
                return f"GPU Memory: {allocated:.2f} GB"
            elif torch.backends.mps.is_available():
                allocated = torch.mps.current_allocated_memory() / (1024**3)  # Convert to GB
                return f"MPS Memory: {allocated:.2f} GB"
            else:
                return "Memory: N/A (CPU)"
        except Exception:
            return "Memory: N/A"

    def on_train_begin(self, args, state, control, **kwargs):
        """Called at the beginning of training."""
        self._trainer = kwargs.get('trainer')  # Store trainer reference
        self.start_time = time.time()
        self.total_steps = state.max_steps
        
        # Print initial training parameters once
        logger.info("\n=== Training Parameters ===")
        logger.info(f"Number of examples: {len(state.train_dataloader)}")
        logger.info(f"Number of Epochs: {args.num_train_epochs}")
        logger.info(f"Batch size per device: {args.per_device_train_batch_size}")
        logger.info(f"Total batch size: {args.per_device_train_batch_size * args.gradient_accumulation_steps}")
        logger.info(f"Gradient Accumulation steps: {args.gradient_accumulation_steps}")
        logger.info(f"Total optimization steps: {state.max_steps}")
        
        # Initialize progress spinner
        self.spinner = ProgressSpinner(
            f"Training - Epoch 0/{int(args.num_train_epochs)}"
        )
        self.spinner.start()

    def on_step_end(self, args, state, control, **kwargs):
        """Called at the end of each step."""
        if state.global_step % args.logging_steps == 0:
            current_epoch = int(state.epoch)
            mem_info = self._get_memory_info()
            
            self.spinner.update_message(
                f"Training - Epoch {current_epoch}/{int(args.num_train_epochs)} | {mem_info}"
            )

    def on_evaluate(self, args, state, control, **kwargs):
        """Called when evaluation starts."""
        if self.spinner:
            self.spinner.update_message(
                f"Evaluating - Epoch {int(state.epoch)}/{int(args.num_train_epochs)}"
            )

    def on_epoch_end(self, args, state, control, **kwargs):
        """Called at the end of each epoch."""
        if self.spinner:
            current_epoch = int(state.epoch)
            self.spinner.update_message(
                f"Completed Epoch {current_epoch}/{int(args.num_train_epochs)} | "
                f"{self._get_memory_info()}"
            )

    def on_train_end(self, args, state, control, **kwargs):
        """Called at the end of training."""
        if self.spinner:
            self.spinner.stop("Training completed")
            
        total_time = time.time() - self.start_time
        
        logger.info("\n=== Training Completed ===")
        logger.info(f"Total training time: {total_time/60:.1f}m")
        logger.info("\nTo try your model:")
        logger.info("1. Interactive mode: python -m src.dialogue --adapter_path output/final_adapter")
        logger.info("2. Compare with base: python compare_models.py")

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Override to prevent default JSON logging."""
        # Don't print the logs directly
        pass