"""CUDA-optimized trainer implementation for efficient training on NVIDIA GPUs."""

import logging
import torch
from torch.amp import autocast, GradScaler
from torch.optim import AdamW
from transformers import Trainer
from typing import Dict, Any, Optional, Union, List, Tuple
from tqdm.auto import tqdm
from .base_optimized_trainer import BaseOptimizedTrainer

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class CUDAOptimizedTrainer(BaseOptimizedTrainer):
    """Trainer optimized for NVIDIA CUDA GPUs.
    
    This trainer extends the HuggingFace Trainer class with optimizations for:
    - Automatic Mixed Precision (AMP)
    - CUDA Graphs
    - Memory management
    - Gradient accumulation
    
    Attributes:
        model: The model to train
        args: The training arguments
        scaler: Gradient scaler for AMP
    """
    
    def __init__(self, **kwargs) -> None:
        """Initialize the CUDA optimized trainer.
        
        Args:
            **kwargs: Keyword arguments passed to the parent Trainer class
        """
        super().__init__(**kwargs)
        self.scaler = GradScaler('cuda')
        self._setup_cuda_optimized_training()

    def _setup_cuda_optimized_training(self) -> None:
        """Configure CUDA-optimized training settings.
        
        This method:
        - Enables automatic mixed precision
        - Sets up CUDA graphs if possible
        - Configures memory allocation
        """
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device not available")
            
        # Set memory allocation strategy
        torch.cuda.set_per_process_memory_fraction(0.95)  # Reserve some memory
        torch.backends.cudnn.benchmark = True
        
        # Enable gradient checkpointing if model supports it
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
            
        logger.info(f"CUDA device: {torch.cuda.get_device_name()}")
        logger.info(f"Available memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    def create_optimizer(self) -> torch.optim.Optimizer:
        """Create an optimized AdamW optimizer.
        
        Returns:
            torch.optim.Optimizer: Configured optimizer instance
        """
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

            self.optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=self.args.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-8
            )
            
            logger.info(f"Created optimizer with learning rate: {self.args.learning_rate}")
        
        return self.optimizer

    def training_step(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, torch.Tensor],
        num_items_in_batch: Optional[int] = None
    ) -> torch.Tensor:
        """Perform a training step with mixed precision.
        
        Args:
            model: The model to train
            inputs: The inputs and targets of the model
            num_items_in_batch: Number of items in the batch
            
        Returns:
            torch.Tensor: The loss value
        """
        model.train()
        inputs = self._prepare_inputs(inputs)
        
        # Update progress bar description with loss
        if hasattr(self, 'progress_bar'):
            current_loss = getattr(self, 'current_loss', 0)
            self.progress_bar.set_description(
                f"Training Loss: {current_loss:.4f}"
            )

        # Mixed precision training step
        with autocast('cuda', dtype=torch.float16):
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
                
            if self.args.gradient_accumulation_steps > 1:
                loss = loss / self.args.gradient_accumulation_steps

        # Scale loss and compute gradients
        self.scaler.scale(loss).backward()
        
        # Store current loss for progress bar and logging
        self.current_loss = loss.detach().float()
        
        # Ensure loss is logged immediately
        self.log({"loss": self.current_loss.item()})
        
        return loss.detach()

    def evaluate(
        self,
        eval_dataset = None,
        ignore_keys = None,
        metric_key_prefix: str = "eval"
    ) -> Dict[str, float]:
        """Override evaluate to implement memory-efficient evaluation."""
        # Clear memory before evaluation
        self._clear_memory()
        
        # Set evaluation batch size to match training batch size
        eval_batch_size = self.args.per_device_train_batch_size
        original_batch_size = self.args.per_device_eval_batch_size
        self.args.per_device_eval_batch_size = eval_batch_size
        
        try:
            # Run evaluation with smaller batch size
            metrics = super().evaluate(
                eval_dataset=eval_dataset,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix
            )
        finally:
            # Restore original batch size
            self.args.per_device_eval_batch_size = original_batch_size
            # Clear memory after evaluation
            self._clear_memory()
            
        return metrics

    def _clear_memory(self) -> None:
        """Comprehensive memory cleanup."""
        # Clear CUDA cache
        torch.cuda.empty_cache()
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Clear any leftover gradients
        if self.model is not None:
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad = None
        
        logger.info(
            f"Memory cleared - Allocated: {torch.cuda.memory_allocated()/1e9:.2f}GB, "
            f"Reserved: {torch.cuda.memory_reserved()/1e9:.2f}GB"
        )

    def train(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        trial: Union[Any, None] = None,
        **kwargs,
    ):
        """Override train to add memory management."""
        # Clear memory before training
        self._clear_memory()
        
        try:
            output = super().train(
                resume_from_checkpoint=resume_from_checkpoint,
                trial=trial,
                **kwargs
            )
            return output
            
        except Exception as e:
            self._clear_memory()  # Clean up on error
            raise e
        finally:
            self._clear_memory()  # Always clean up after training

    def _maybe_log_save_evaluate(self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time):
        """Override to add memory management around evaluation."""
        # Clear memory before evaluation
        self._clear_memory()
        
        try:
            super()._maybe_log_save_evaluate(
                tr_loss, grad_norm, model, trial, epoch, 
                ignore_keys_for_eval, start_time
            )
        finally:
            # Clear memory after evaluation
            self._clear_memory()

    def _save_checkpoint(
        self, 
        model: torch.nn.Module,
        trial: Union[None, Any],
        metrics: Optional[Dict[str, float]] = None
    ) -> None:
        """Save a checkpoint during training."""
        # Empty CUDA cache before saving
        torch.cuda.empty_cache()
        
        try:
            super()._save_checkpoint(model, trial)
            if metrics:
                logger.info(f"Checkpoint saved with metrics: {metrics}")
            else:
                logger.info("Checkpoint saved successfully")
        except Exception as e:
            logger.error(f"Error saving checkpoint: {str(e)}")
            raise

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        """Enhanced logging with progress bar updates and CUDA metrics."""
        if hasattr(self, 'progress_bar'):
            # Update progress bar with current metrics
            desc_items = []
            
            # Add CUDA memory info
            cuda_memory = torch.cuda.memory_allocated() / 1e9
            desc_items.append(f"CUDA Mem: {cuda_memory:.1f}GB")
            
            # Add other metrics
            for k, v in logs.items():
                if isinstance(v, (int, float)):
                    desc_items.append(f"{k}: {v:.4f}")
            
            if desc_items:
                self.progress_bar.set_description(" | ".join(desc_items))
            
            # Update progress
            if "epoch" in logs:
                self.progress_bar.update(1)
        
        # Add CUDA memory metrics to logs
        logs["cuda_memory_allocated"] = torch.cuda.memory_allocated() / 1e9
        logs["cuda_memory_reserved"] = torch.cuda.memory_reserved() / 1e9
        
        super().log(logs, start_time)

    def _setup_device_specific_training(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device not available")
        torch.backends.cudnn.benchmark = True
        self.scaler = GradScaler('cuda')
        
    def _optimize_memory_allocation(self) -> None:
        torch.cuda.set_per_process_memory_fraction(0.95)
        
    def _clear_device_cache(self) -> None:
        torch.cuda.empty_cache()
