"""Training progress callback module for monitoring training progress."""

import logging
import torch
import time
from transformers import TrainerCallback, TrainingArguments
from typing import Dict, Any, Optional
from tqdm.auto import tqdm
import sys

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class ProgressSpinner:
    """Simple spinner animation for indeterminate progress."""
    def __init__(self, message: str):
        self.message = message
        self.frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self.current = 0
        self.active = True
        
    def spin(self):
        """Update spinner frame."""
        if self.active:
            frame = self.frames[self.current]
            sys.stdout.write(f'\r{frame} {self.message}')
            sys.stdout.flush()
            self.current = (self.current + 1) % len(self.frames)
            
    def stop(self, final_message: Optional[str] = None):
        """Stop spinner and optionally show final message."""
        self.active = False
        if final_message:
            sys.stdout.write(f'\r✓ {final_message}\n')
        else:
            sys.stdout.write('\r✓ Done!\n')
        sys.stdout.flush()

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
        self.start_time = None
        self.epoch_start_time = None
        self.total_steps = 0
        self.current_phase = None
        self.phases = [
            "Loading Model", 
            "Processing Dataset",
            "Training",
            "Evaluation",
            "Saving Adapters"
        ]
        # Detect device type once at initialization
        self.device_type = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        
    def _get_memory_info(self) -> str:
        """Get memory usage based on device type."""
        try:
            if self.device_type == "cuda":
                allocated = torch.cuda.memory_allocated() / (1024**2)
                return f"GPU Memory: {allocated:.2f} MB"
            elif self.device_type == "mps":
                allocated = torch.mps.current_allocated_memory() / (1024**2)
                return f"MPS Memory: {allocated:.2f} MB"
            else:
                return "Memory: N/A (CPU)"
        except Exception:
            return "Memory: N/A"

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: Dict[str, Any],
        control: Dict[str, Any],
        **kwargs
    ) -> None:
        """Called at the beginning of training."""
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

    def _get_current_loss(self, state: Dict[str, Any]) -> Optional[float]:
        """Safely get the current loss value.
        
        Args:
            state: Training state dict
            
        Returns:
            Optional[float]: Current loss value or None if not available
        """
        try:
            if state.log_history:
                return state.log_history[-1].get('loss')
            return None
        except (IndexError, AttributeError):
            return None

    def on_step_end(
        self,
        args: TrainingArguments,
        state: Dict[str, Any],
        control: Dict[str, Any],
        **kwargs
    ) -> None:
        """Called at the end of each step."""
        if state.global_step % args.logging_steps == 0:
            # Update spinner message with current epoch and memory info
            current_epoch = int(state.epoch)
            mem_info = self._get_memory_info()
            self.spinner.message = (
                f"Training - Epoch {current_epoch}/{int(args.num_train_epochs)} | "
                f"{mem_info} | Loss: {state.log_history[-1].get('loss', 'N/A'):.4f}"
            )

    def on_epoch_begin(
        self,
        args: TrainingArguments,
        state: Dict[str, Any],
        control: Dict[str, Any],
        **kwargs
    ) -> None:
        """Called at the beginning of each epoch."""
        self.epoch_start_time = time.time()
        logger.info(f"\n=== Starting Epoch {int(state.epoch) + 1}/{int(args.num_train_epochs)} ===")

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: Dict[str, Any],
        control: Dict[str, Any],
        **kwargs
    ) -> None:
        """Called at the end of each epoch."""
        current_epoch = int(state.epoch)
        self.spinner.message = (
            f"Completed Epoch {current_epoch}/{int(args.num_train_epochs)} | "
            f"{self._get_memory_info()}"
        )

    def on_train_end(
        self,
        args: TrainingArguments,
        state: Dict[str, Any],
        control: Dict[str, Any],
        **kwargs
    ) -> None:
        """Called at the end of training."""
        total_time = time.time() - self.start_time
        
        logger.info("\n=== Training Completed ===")
        logger.info(f"Total training time: {total_time/60:.1f}m")
        logger.info("\nTo try your model:")
        logger.info("1. Interactive mode: python -m src.dialogue --adapter_path output/final_adapter")
        logger.info("2. Compare with base: python compare_models.py")
        
        if torch.backends.mps.is_available():
            logger.info(f"Final Memory Usage: {torch.mps.current_allocated_memory() / 1024**2:.2f} MB")

    def on_evaluate(self, args, state, control, **kwargs):
        """Called when evaluation starts."""
        self.spinner.message = f"Evaluating - Epoch {int(state.epoch)}/{int(args.num_train_epochs)}"

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Override to prevent default JSON logging."""
        # Don't print the logs directly
        pass