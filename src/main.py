"""Main entry point for Phi-2 fine-tuning with LoRA."""

import argparse
import logging
import os
import sys
import torch

from src.config.training_config import TrainingConfig
from src.trainers.phi2_lora_trainer import Phi2LoRATrainer
from src.utils.exceptions import (
    DatasetValidationError,
    ModelPreparationError,
    DeviceError,
    TrainingError
)


# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def setup_arg_parser() -> argparse.ArgumentParser:
    """Set up command line argument parser.
    
    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="Fine-tune Phi-2 with LoRA for humorous responses"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the model"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the JSONL dataset file"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Training batch size per device"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=32,
        help="Number of gradient accumulation steps"
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.03,
        help="Warmup ratio for learning rate scheduler"
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=8,
        help="LoRA attention dimension"
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha parameter"
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,
        help="LoRA dropout probability"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    return parser

def validate_environment() -> None:
    """Validate the execution environment.
    
    Raises:
        DeviceError: If MPS is not available
    """
    if not torch.backends.mps.is_available():
        raise DeviceError(
            "MPS not available. Ensure macOS 12.3+ and PyTorch MPS support"
        )
    logger.info("MPS (Metal Performance Shaders) is available")

def setup_training_config(args: argparse.Namespace) -> TrainingConfig:
    """Create training configuration from command line arguments.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        TrainingConfig: Configuration object for training
    """
    return TrainingConfig(
        batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        max_seq_length=args.max_length,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout
    )

def main() -> None:
    """Main execution function."""
    parser = setup_arg_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Validate environment
        validate_environment()

        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Setup training configuration
        config = setup_training_config(args)

        # Initialize trainer
        trainer = Phi2LoRATrainer(
            output_dir=args.output_dir,
            dataset_path=args.dataset_path,
            config=config
        )

        logger.info("Preparing model...")
        trainer.prepare_model()

        logger.info("Starting training...")
        trainer.train()

        logger.info(f"Training completed. Model saved to {args.output_dir}")

    except DatasetValidationError as e:
        logger.error(f"Dataset validation failed: {str(e)}")
        sys.exit(1)
    except ModelPreparationError as e:
        logger.error(f"Model preparation failed: {str(e)}")
        sys.exit(1)
    except DeviceError as e:
        logger.error(f"Device error: {str(e)}")
        sys.exit(1)
    except TrainingError as e:
        logger.error(f"Training error: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise

if __name__ == "__main__":
    main()