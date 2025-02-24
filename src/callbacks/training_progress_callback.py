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
        
        logger.info("\n=== Training Pipeline Started ===")
        logger.info("Total epochs: %d", args.num_train_epochs)
        logger.info("Training steps per epoch: %d", state.max_steps // args.num_train_epochs)
        logger.info("Batch size: %d", args.per_device_train_batch_size)
        
        if torch.backends.mps.is_available():
            logger.debug(f"Initial MPS Memory: {torch.mps.current_allocated_memory() / 1024**2:.2f} MB")

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
            # Calculate progress and timing
            elapsed_time = time.time() - self.start_time
            steps_per_second = state.global_step / elapsed_time if elapsed_time > 0 else 0
            remaining_steps = self.total_steps - state.global_step
            eta = remaining_steps / steps_per_second if steps_per_second > 0 else 0
            
            # Get current loss
            current_loss = self._get_current_loss(state)
            loss_info = f"Loss: {current_loss:.4f}" if current_loss is not None else "Loss: N/A"
            
            # Basic progress info
            logger.info(
                f"Step [{state.global_step}/{self.total_steps}] | "
                f"{loss_info} | "
                f"Speed: {steps_per_second:.1f} steps/s | "
                f"ETA: {eta/60:.1f}m"
            )
            
            # Detailed metrics at DEBUG level
            logger.debug(
                f"Epoch: {state.epoch:.2f} | "
                f"Memory: {torch.mps.current_allocated_memory() / 1024**2:.2f} MB"
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
        epoch_time = time.time() - self.epoch_start_time
        eval_loss = self._get_current_loss(state)
        
        logger.info(
            f"Completed Epoch {int(state.epoch)}/{int(args.num_train_epochs)} "
            f"in {epoch_time/60:.1f}m | "
            f"Loss: {eval_loss:.4f}" if eval_loss is not None else "N/A"
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