"""Main entry point for Phi-2 fine-tuning with LoRA."""

import argparse
import logging
import os
import sys
import torch
import json

from src.config.training_config import TrainingConfig
from src.trainers.phi2_lora_trainer import Phi2LoRATrainer
from src.utils.exceptions import (
    DatasetValidationError,
    ModelPreparationError,
    DeviceError,
    TrainingError
)
from src.utils.device_utils import detect_device


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
        description="Fine-tune language models with LoRA for humorous responses"
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
        default=4,
        help="Training batch size per device"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
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
        default=16,
        help="Number of gradient accumulation steps"
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.05,
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
        default=0.05,
        help="LoRA dropout probability"
    )
    parser.add_argument(
        "--verbose",
        type=str,
        choices=["none", "info", "debug"],
        default="info",
        help="Verbosity level (none, info, debug)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="microsoft/phi-2",
        help="Name/path of the model to use (default: microsoft/phi-2)"
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        default=True,
        help="Enable gradient checkpointing to save memory"
    )
    parser.add_argument(
        "--torch_compile",
        action="store_true",
        default=True,
        help="Enable torch.compile for optimization"
    )
    return parser

def validate_environment() -> None:
    """Validate the execution environment.
    
    Raises:
        DeviceError: If no suitable device is available
    """
    device_type, device_str = detect_device()
    
    if device_type == "cuda":
        if not torch.cuda.is_available():
            raise DeviceError("CUDA device detected but not available")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
    elif device_type == "mps":
        if not torch.backends.mps.is_available():
            raise DeviceError("MPS not available. Ensure macOS 12.3+ and PyTorch MPS support")
        logger.info("Using MPS (Metal Performance Shaders)")
    else:
        logger.warning("No GPU detected, using CPU. Training may be slow")

def validate_dataset_file(dataset_path: str) -> None:
    """Validate that the dataset file exists and has valid JSONL format.
    
    Args:
        dataset_path: Path to the dataset file
        
    Raises:
        DatasetValidationError: If validation fails
    """
    # Check if file exists
    if not os.path.exists(dataset_path):
        raise DatasetValidationError(
            f"Dataset file not found: {dataset_path}\n"
            "Please check:\n"
            "1. The file path is correct\n"
            "2. You have read permissions for the file\n"
            "3. The file exists in the specified location"
        )
    
    # Check file extension
    if not dataset_path.endswith('.jsonl'):
        raise DatasetValidationError(
            f"Invalid file format: {dataset_path}\n"
            "The dataset must be a JSONL file with .jsonl extension.\n"
            "Each line should contain a JSON object with 'prompt' and 'response' fields."
        )
    
    # Validate JSONL format and required fields
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            first_line = False
            line_number = 0
            for line in f:
                line_number += 1
                try:
                    data = json.loads(line.strip())
                    if not first_line:
                        if not isinstance(data, dict):
                            raise DatasetValidationError(
                                f"Invalid data format at line {line_number}. "
                                "Each line must contain a JSON object."
                            )
                        if 'prompt' not in data or 'response' not in data:
                            raise DatasetValidationError(
                                f"Missing required fields at line {line_number}.\n"
                                "Each line must contain 'prompt' and 'response' fields.\n"
                                f"Found fields: {list(data.keys())}"
                            )
                        first_line = True
                except json.JSONDecodeError as e:
                    raise DatasetValidationError(
                        f"Invalid JSON at line {line_number}: {str(e)}\n"
                        "Each line must be a valid JSON object."
                    )
            
            if line_number == 0:
                raise DatasetValidationError(
                    f"Empty dataset file: {dataset_path}\n"
                    "The JSONL file must contain at least one example."
                )
            
            logger.info(f"Dataset validation successful: {line_number} examples found")
            
    except Exception as e:
        if isinstance(e, DatasetValidationError):
            raise e
        raise DatasetValidationError(
            f"Error reading dataset file: {str(e)}\n"
            "Please ensure the file is accessible and properly formatted."
        )

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

def setup_logging(verbose_level: str) -> None:
    """Configure logging based on verbosity level."""
    if verbose_level == "none":
        logging.getLogger().setLevel(logging.WARNING)
    elif verbose_level == "info":
        logging.getLogger().setLevel(logging.INFO)
    elif verbose_level == "debug":
        logging.getLogger().setLevel(logging.DEBUG)

def main() -> None:
    """Main execution function."""
    parser = setup_arg_parser()
    args = parser.parse_args()
    
    print("") # Just formatting

    # Configure logging based on verbosity
    setup_logging(args.verbose)

    try:
        # Validate environment
        validate_environment()
        
        # Validate dataset file first
        validate_dataset_file(args.dataset_path)

        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Setup training configuration
        config = setup_training_config(args)

        # Initialize trainer
        trainer = Phi2LoRATrainer(
            output_dir=args.output_dir,
            dataset_path=args.dataset_path,
            model_name=args.model_name,
            config=config
        )

        logger.info("Preparing model...")
        trainer.prepare_model()

        logger.info("Starting training...")
        trainer.train()

        logger.info(f"Training completed. Model saved to {args.output_dir}")

    except DatasetValidationError as e:
        logger.error("Dataset Validation Error:")
        logger.error("------------------------")
        logger.error(str(e))
        logger.error("\nPlease fix the dataset issues and try again.")
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