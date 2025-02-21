"""Training progress callback module for monitoring training progress."""

import logging
import torch
from transformers import TrainerCallback, TrainingArguments
from typing import Dict, Any, Optional

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

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: Dict[str, Any],
        control: Dict[str, Any],
        **kwargs
    ) -> None:
        """Called at the beginning of training."""
        logger.info("\n=== Starting Training ===")
        if torch.backends.mps.is_available():
            logger.info(
                f"Initial MPS Memory: {torch.mps.current_allocated_memory() / 1024**2:.2f} MB"
            )

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
            # Calculate training progress
            epoch = state.epoch or 0
            steps_per_epoch = (
                state.max_steps // args.num_train_epochs 
                if args.num_train_epochs > 0 
                else 0
            )
            current_step = (
                state.global_step % steps_per_epoch 
                if steps_per_epoch > 0 
                else 0
            )
            progress = (
                (current_step / steps_per_epoch * 100) 
                if steps_per_epoch > 0 
                else 0
            )

            # Get current loss safely
            current_loss = self._get_current_loss(state)
            loss_info = f"Loss: {current_loss:.4f}" if current_loss is not None else "Loss: N/A"

            # Log progress information
            logger.info(
                f"Epoch: {epoch:.2f} | "
                f"Step: {state.global_step} | "
                f"Progress: {progress:.1f}% | "
                f"{loss_info}"
            )

            # Log memory usage if MPS is available
            if torch.backends.mps.is_available():
                memory = torch.mps.current_allocated_memory() / 1024**2
                logger.info(f"MPS Memory: {memory:.2f} MB")

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: Dict[str, Any],
        control: Dict[str, Any],
        **kwargs
    ) -> None:
        """Called at the end of each epoch."""
        logger.info(f"\n=== Completed Epoch {int(state.epoch)} ===")
        
        eval_loss = self._get_current_loss(state)
        if eval_loss is not None:
            logger.info(f"Evaluation Loss: {eval_loss:.4f}")

    def on_train_end(
        self,
        args: TrainingArguments,
        state: Dict[str, Any],
        control: Dict[str, Any],
        **kwargs
    ) -> None:
        """Called at the end of training."""
        logger.info("\n=== Training Completed ===")
        if torch.backends.mps.is_available():
            logger.info(
                f"Final MPS Memory: {torch.mps.current_allocated_memory() / 1024**2:.2f} MB"
            )