"""Main trainer class for fine-tuning Phi-2 with LoRA."""

import os
import shutil
import logging
import torch
from typing import Optional, Dict, Any, Tuple
from datasets import load_dataset, Dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType

from config.training_config import TrainingConfig
from utils.exceptions import DatasetValidationError, ModelPreparationError, DeviceError
from trainers.mps_optimized_trainer import MPSOptimizedTrainer
from callbacks.training_progress_callback import TrainingProgressCallback

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class Phi2LoRATrainer:
    """Main trainer class for Phi-2 with LoRA.
    
    This class handles:
    - Model and tokenizer initialization
    - Dataset preparation and validation
    - LoRA configuration
    - Training process management
    
    Attributes:
        model_name (str): Name of the base model
        output_dir (str): Directory for saving outputs
        dataset_path (str): Path to the dataset file
        config (TrainingConfig): Training configuration
        model: The model instance
        tokenizer: The tokenizer instance
        device: The training device
        lora_config: LoRA configuration
    """

    def __init__(
        self,
        output_dir: str,
        dataset_path: str,
        config: Optional[TrainingConfig] = None
    ) -> None:
        """Initialize the trainer.
        
        Args:
            output_dir: Directory to save the model
            dataset_path: Path to the JSONL dataset file
            config: Optional training configuration
        """
        self.model_name = "microsoft/phi-2"
        self.output_dir = output_dir
        self.dataset_path = dataset_path
        self.config = config or TrainingConfig()
        self._setup_model_and_tokenizer()

    def _setup_model_and_tokenizer(self) -> None:
        """Initialize model and tokenizer with MPS-optimized settings."""
        try:
            # Check MPS availability
            if not torch.backends.mps.is_available():
                logger.warning("MPS not available. Falling back to CPU. This will be slow!")
                self.device = torch.device("cpu")
            else:
                self.device = torch.device("mps")
                logger.info("Using MPS (Metal Performance Shaders) for acceleration")

            # Initialize tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                padding_side="right"
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with float32 for MPS compatibility
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                device_map=None,
                trust_remote_code=True
            ).to(self.device)

            # Enable gradient checkpointing for memory efficiency
            self.model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
            
            # Configure LoRA
            self.lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
                target_modules=["q_proj", "k_proj", "v_proj", "dense"],
                inference_mode=False
            )
            logger.info("LoRA configuration completed")
            
        except Exception as e:
            raise ModelPreparationError(f"Failed to setup model and tokenizer: {str(e)}")

    def validate_dataset(self, dataset: Dict[str, Any]) -> bool:
        """Validate dataset structure and content.
        
        Args:
            dataset: Dataset to validate
            
        Returns:
            bool: True if validation successful
            
        Raises:
            DatasetValidationError: If validation fails
        """
        if not dataset or "train" not in dataset:
            raise DatasetValidationError("Invalid dataset structure")

        example = dataset['train'][0]
        required_fields = ['prompt', 'response']
        missing_fields = [f for f in required_fields if f not in example]

        if missing_fields:
            raise DatasetValidationError(
                f"Missing fields: {missing_fields}. Found: {list(example.keys())}"
            )

        logger.info(f"Dataset validation successful: {len(dataset['train'])} examples")
        return True

    def prepare_model(self) -> None:
        """Prepare model with LoRA configuration."""
        try:
            if torch.backends.mps.is_available():
                logger.info(
                    f"Memory allocated: {torch.mps.current_allocated_memory() / 1024**2:.2f} MB"
                )
            
            self.model = get_peft_model(self.model, self.lora_config)
            
            trainable_params = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )
            total_params = sum(p.numel() for p in self.model.parameters())
            
            logger.info(
                f"Trainable parameters: {trainable_params:,} "
                f"({trainable_params/total_params:.2%})"
            )
            
            if torch.backends.mps.is_available():
                logger.info(
                    f"Memory allocated after prep: "
                    f"{torch.mps.current_allocated_memory() / 1024**2:.2f} MB"
                )
            
        except Exception as e:
            raise ModelPreparationError(f"Error preparing model: {str(e)}")

    @staticmethod
    def format_instruction(prompt: str, response: str) -> str:
        """Format the instruction template.
        
        Args:
            prompt: Input prompt
            response: Expected response
            
        Returns:
            str: Formatted instruction
        """
        return (
            "Below is an instruction that describes a task. Write a response that "
            "completes the request in a humorous and engaging way.\n\n"
            "### Instruction:\n"
            f"{prompt}\n\n"
            "### Response:\n"
            f"{response}\n"
        )

    def prepare_dataset(self) -> DatasetDict:
        """Prepare and tokenize the dataset.
        
        Returns:
            DatasetDict: Processed and split dataset
            
        Raises:
            DatasetValidationError: If dataset preparation fails
        """
        try:
            dataset = load_dataset('json', data_files=self.dataset_path)
            self.validate_dataset(dataset)

            def tokenize_function(examples: Dict[str, Any]) -> Dict[str, Any]:
                """Tokenize the examples."""
                texts = [
                    self.format_instruction(p, r)
                    for p, r in zip(examples['prompt'], examples['response'])
                ]
                
                # Tokenize with padding and truncation
                encoded = self.tokenizer(
                    texts,
                    truncation=True,
                    max_length=self.config.max_seq_length,
                    padding="max_length",
                    return_tensors=None  # Return as lists, not tensors
                )
                
                # Create labels for causal language modeling
                encoded["labels"] = encoded["input_ids"].copy()
                
                return encoded

            tokenized_dataset = dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=dataset["train"].column_names
            )

            splits = tokenized_dataset["train"].train_test_split(
                test_size=0.1,
                shuffle=True,
                seed=42
            )
            return splits

        except Exception as e:
            raise DatasetValidationError(f"Dataset preparation failed: {str(e)}")

    def _setup_training_arguments(self, use_tensorboard: bool) -> TrainingArguments:
        """Set up training arguments.
        
        Args:
            use_tensorboard: Whether to use tensorboard for logging
            
        Returns:
            TrainingArguments: Configured training arguments
        """
        return TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            max_grad_norm=self.config.max_grad_norm,
            warmup_ratio=self.config.warmup_ratio,
            logging_dir=os.path.join(self.output_dir, "logs"),
            logging_strategy="steps",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_steps=1,
            logging_first_step=True,
            report_to=["tensorboard"] if use_tensorboard else [],
            disable_tqdm=False,
            save_total_limit=1,
            optim="adamw_torch",
            fp16=False,
            bf16=False,
            remove_unused_columns=False,
            gradient_checkpointing=True,
            weight_decay=0.01,
            load_best_model_at_end=True,
            push_to_hub=False,
            hub_strategy="end",
            hub_model_id=None,
            save_safetensors=True,
            log_level="info",
            log_on_each_node=True,
            logging_nan_inf_filter=True
        )

    def train(self) -> None:
        """Execute the training process."""
        # Check tensorboard availability
        try:
            import tensorboard
            use_tensorboard = True
        except ImportError:
            logger.warning(
                "Tensorboard not installed. Training will proceed without tensorboard logging."
            )
            use_tensorboard = False

        training_args = self._setup_training_arguments(use_tensorboard)

        try:
            # Prepare dataset
            tokenized_dataset = self.prepare_dataset()

            # Create data collator for language modeling
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False  # We want causal language modeling, not masked
            )

            # Initialize trainer
            trainer = MPSOptimizedTrainer(
                model=self.model,
                args=training_args,
                train_dataset=tokenized_dataset["train"],
                eval_dataset=tokenized_dataset["test"],
                data_collator=data_collator,
                callbacks=[TrainingProgressCallback()]
            )

            # Start training
            trainer.train()
            logger.info("Training completed")
            
            # Save only the LoRA adapter weights
            final_checkpoint_path = os.path.join(self.output_dir, "final_adapter")
            self.model.save_pretrained(final_checkpoint_path, safe_serialization=True)
            logger.info(f"Model saved to {final_checkpoint_path}")
            
            # Clean up intermediate checkpoints
            for item in os.listdir(self.output_dir):
                if item.startswith("checkpoint-"):
                    shutil.rmtree(os.path.join(self.output_dir, item))
                    logger.info(f"Cleaned up checkpoint: {item}")
            
            # Report size
            adapter_size = sum(
                os.path.getsize(os.path.join(final_checkpoint_path, f))
                for f in os.listdir(final_checkpoint_path)
            )
            logger.info(f"Final adapter size: {adapter_size / (1024*1024):.2f} MB")

        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise