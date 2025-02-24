"""CUDA-optimized trainer implementation for efficient training on NVIDIA GPUs."""

import logging
import torch
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from transformers import Trainer
from typing import Dict, Any, Optional, Union, List, Tuple
from tqdm.auto import tqdm
from trainers.base_optimized_trainer import BaseOptimizedTrainer

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
        self.scaler = GradScaler()
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
        with autocast(device_type='cuda', dtype=torch.float16):
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
                
            if self.args.gradient_accumulation_steps > 1:
                loss = loss / self.args.gradient_accumulation_steps

        # Scale loss and compute gradients
        self.scaler.scale(loss).backward()
        
        # Store current loss for progress bar
        self.current_loss = loss.detach().float()
        
        return loss.detach()

    def train(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        trial: Union[Any, None] = None,
        **kwargs,
    ):
        """Override train to add progress bar and CUDA-specific optimizations."""
        # Calculate total steps
        total_steps = len(self.train_dataset) * self.args.num_train_epochs
        
        # Create progress bar
        self.progress_bar = tqdm(
            total=total_steps,
            desc="Training",
            position=0,
            leave=True,
            dynamic_ncols=True
        )
        
        try:
            # Empty CUDA cache before training
            torch.cuda.empty_cache()
            
            output = super().train(resume_from_checkpoint=resume_from_checkpoint, trial=trial, **kwargs)
            self.progress_bar.close()
            return output
            
        except Exception as e:
            self.progress_bar.close()
            raise e

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
        self.scaler = GradScaler()
        
    def _optimize_memory_allocation(self) -> None:
        torch.cuda.set_per_process_memory_fraction(0.95)
        
    def _clear_device_cache(self) -> None:
        torch.cuda.empty_cache()
