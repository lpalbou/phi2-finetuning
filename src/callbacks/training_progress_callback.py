"""Training progress callback module using Rich progress bars."""

import logging
import torch
from transformers import TrainerCallback
from typing import Dict, Optional
from .progress_display import ProgressDisplay

# Configure logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainingProgressCallback(TrainerCallback):
    """Rich-based training progress callback."""
    
    def __init__(self):
        """Initialize the callback."""
        super().__init__()
        self.progress = ProgressDisplay()
        self.start_time = None
        self._trainer = None
        
    def _get_memory_info(self) -> str:
        """Get memory usage string."""
        try:
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / (1024**3)
                return f"GPU: {allocated:.2f}GB"
            elif torch.backends.mps.is_available():
                allocated = torch.mps.current_allocated_memory() / (1024**3)
                return f"MPS: {allocated:.2f}GB"
            return "CPU"
        except Exception:
            return "N/A"

    def on_train_begin(self, args, state, control, **kwargs):
        """Start training progress display."""
        self._trainer = kwargs.get('trainer')
        description = f"Training - Epoch 0/{int(args.num_train_epochs)}"
        self.progress.start(description, total=state.max_steps)
        
        # Log training parameters
        logger.info("\n=== Training Parameters ===")
        logger.info(f"Examples: {len(state.train_dataloader)}")
        logger.info(f"Epochs: {args.num_train_epochs}")
        logger.info(f"Batch size: {args.per_device_train_batch_size}")
        logger.info(f"Total batch size: {args.per_device_train_batch_size * args.gradient_accumulation_steps}")
        logger.info(f"Gradient Accumulation steps: {args.gradient_accumulation_steps}")
        logger.info(f"Total optimization steps: {state.max_steps}")

    def on_step_end(self, args, state, control, **kwargs):
        """Update progress on step end."""
        if state.global_step % args.logging_steps == 0:
            current_epoch = int(state.epoch)
            mem_info = self._get_memory_info()
            description = f"Training - Epoch {current_epoch}/{int(args.num_train_epochs)} | {mem_info}"
            self.progress.update(description=description, advance=args.logging_steps)

    def on_evaluate(self, args, state, control, **kwargs):
        """Update display during evaluation."""
        description = f"Evaluating - Epoch {int(state.epoch)}/{int(args.num_train_epochs)}"
        self.progress.update(description=description)

    def on_epoch_end(self, args, state, control, **kwargs):
        """Update display on epoch end."""
        current_epoch = int(state.epoch)
        mem_info = self._get_memory_info()
        description = f"Completed Epoch {current_epoch}/{int(args.num_train_epochs)} | {mem_info}"
        self.progress.update(description=description)

    def on_train_end(self, args, state, control, **kwargs):
        """Handle end of training."""
        self.progress.stop("Training completed")
        
        logger.info("\n=== Training Completed ===")
        logger.info("\nTo try your model:")
        logger.info("1. Interactive mode: python -m src.dialogue --adapter_path output/final_adapter")
        logger.info("2. Compare with base: python compare_models.py")