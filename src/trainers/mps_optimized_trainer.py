"""MPS-optimized trainer implementation for efficient training on Apple Silicon."""

import logging
import torch
from torch.optim import AdamW
from transformers import Trainer
from tqdm.auto import tqdm
from typing import Dict, Any, Optional, Union, List, Tuple
from .base_optimized_trainer import BaseOptimizedTrainer

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class MPSOptimizedTrainer(BaseOptimizedTrainer):
    """Trainer optimized for MPS (Metal Performance Shaders) backend.
    
    This trainer extends the HuggingFace Trainer class with optimizations for:
    - Memory management on MPS devices
    - Optimizer configurations
    - Loss computation
    
    Attributes:
        model: The model to train
        args: The training arguments
        optimizer: The optimizer instance
    """
    
    def __init__(self, **kwargs) -> None:
        """Initialize the MPS optimized trainer.
        
        Args:
            **kwargs: Keyword arguments passed to the parent Trainer class
        """
        super().__init__(**kwargs)

    def create_optimizer(self) -> torch.optim.Optimizer:
        """Create an optimized AdamW optimizer.
        
        Returns:
            torch.optim.Optimizer: Configured optimizer instance
        
        This method creates an optimizer with:
        - Custom parameter grouping
        - Weight decay configuration
        - Learning rate settings
        """
        if self.optimizer is None:
            model = self.model
            no_decay = ["bias", "LayerNorm.weight"]
            
            # Group parameters for optimization
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

            # Create AdamW optimizer
            self.optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=self.args.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-8
            )
            
            logger.info(f"Created optimizer with learning rate: {self.args.learning_rate}")
        
        return self.optimizer

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """Compute training loss with memory optimization.
        
        Args:
            model: The model to train
            inputs: The inputs and targets of the model
            return_outputs: If True, will also return model outputs
            num_items_in_batch: Number of items in the batch (added for compatibility)
        
        Returns:
            torch.Tensor or tuple: Loss value if return_outputs=False,
                                 tuple of (loss, outputs) if return_outputs=True
        """
        # Clear MPS cache if available
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        # Ensure inputs are properly formatted
        if "labels" not in inputs:
            logger.warning("No labels found in inputs. This might cause issues with loss computation.")

        # Compute loss using parent class implementation
        return super().compute_loss(model, inputs, return_outputs)

    def training_step(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, torch.Tensor],
        num_items_in_batch: Optional[int] = None
    ) -> torch.Tensor:
        """Perform a training step with progress tracking."""
        model.train()
        inputs = self._prepare_inputs(inputs)
        
        # Update progress bar description with loss
        if hasattr(self, 'progress_bar'):
            current_loss = getattr(self, 'current_loss', 0)
            self.progress_bar.set_description(
                f"Training Loss: {current_loss:.4f}"
            )
        
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
            
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
            
        loss.backward()
        
        # Store current loss for progress bar
        self.current_loss = loss.detach().float()
        
        return loss.detach()

    def train(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        trial: Union[Any, None] = None,
        **kwargs,
    ):
        """Override train to add progress bar."""
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
        """Save a checkpoint during training.
        
        Args:
            model: The model to save
            trial: A trial instance (for hyperparameter search)
            metrics: Optional dictionary of metric values
        """
        # Clear memory before saving
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        
        # Save checkpoint using parent implementation
        try:
            super()._save_checkpoint(model, trial)
            if metrics:
                logger.info(f"Checkpoint saved with metrics: {metrics}")
            else:
                logger.info("Checkpoint saved successfully")
        except Exception as e:
            logger.error(f"Error saving checkpoint: {str(e)}")
            raise

    def get_train_dataloader(self) -> torch.utils.data.DataLoader:
        """Get train dataloader with MPS optimizations.
        
        Returns:
            DataLoader: Configured train dataloader
        """
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        return super().get_train_dataloader()

    def get_eval_dataloader(self, eval_dataset = None) -> torch.utils.data.DataLoader:
        """Get evaluation dataloader with MPS optimizations.
        
        Args:
            eval_dataset: Optional evaluation dataset
            
        Returns:
            DataLoader: Configured evaluation dataloader
        """
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        return super().get_eval_dataloader(eval_dataset)

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        """Enhanced logging with progress bar updates."""
        if hasattr(self, 'progress_bar'):
            # Update progress bar with current metrics
            desc_items = []
            for k, v in logs.items():
                if isinstance(v, (int, float)):
                    desc_items.append(f"{k}: {v:.4f}")
            
            if desc_items:
                self.progress_bar.set_description(" | ".join(desc_items))
            
            # Update progress
            if "epoch" in logs:
                self.progress_bar.update(1)
        
        super().log(logs, start_time)

    def _setup_device_specific_training(self) -> None:
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS device not available")
            
    def _optimize_memory_allocation(self) -> None:
        # MPS-specific memory optimizations
        pass
        
    def _clear_device_cache(self) -> None:
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()