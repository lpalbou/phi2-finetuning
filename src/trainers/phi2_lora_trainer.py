"""Main trainer class for fine-tuning Phi-2 with LoRA."""

import os
import shutil
import logging
import torch
from typing import Optional, Dict, Any, Tuple, Type
from datasets import load_dataset, Dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from tqdm.auto import tqdm

from ..config.training_config import TrainingConfig
from ..utils.exceptions import DatasetValidationError, ModelPreparationError, DeviceError
from .mps_optimized_trainer import MPSOptimizedTrainer
from ..callbacks.training_progress_callback import TrainingProgressCallback
from .base_optimized_trainer import BaseOptimizedTrainer
from ..utils.device_utils import detect_device
from ..callbacks.progress_display import ProgressDisplay

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

    # Class-level mapping of model configurations
    # q_proj, k_proj, v_proj : key to encoding attention mechanism (how the model attends to different input parts)
    # dense : Processes output from attention layers, refining knowledge representation
    # embed_tokens : how words are represented internally (Defines word embeddings, necessary for learning new terminology)
    # mlp.fc1, mlp.fc2 : key to encoding deeper semantic understanding (Core feed-forward layers that process relationships between learned facts)
    # lm_head : ensure concepts are properly expressed (Controls final text generation, affecting style, fluency, and formatting)
    MODEL_CONFIGS = {
        "microsoft/phi-2": {
            "target_modules": [
                "q_proj", "k_proj", "v_proj", "dense",
                "mlp.fc1", "mlp.fc2", 
                "lm_head", "embed_tokens"
            ],
            "description": "Original Phi-2 model target modules focusing on attention and MLP layers"
        },
        "microsoft/Phi-3.5-mini-instruct": {
            "target_modules": [
                "self_attn.qkv_proj",
                "self_attn.o_proj",
                "mlp.gate_up_proj",
                "mlp.down_proj"
            ],
            "description": "Phi-3.5 mini instruct model target modules targeting attention and MLP layers"
        }
    }

    def __init__(
        self,
        output_dir: str,
        dataset_path: str,
        model_name: str = "microsoft/phi-2",
        config: Optional[TrainingConfig] = None
    ) -> None:
        """Initialize the trainer.
        
        Args:
            output_dir: Directory to save the model
            dataset_path: Path to the JSONL dataset file
            model_name: Name/path of the model to use (default: "microsoft/phi-2")
            config: Optional training configuration
        """
        self.model_name = model_name
        self.output_dir = output_dir
        self.dataset_path = dataset_path
        self.config = config or TrainingConfig()
        
        # Detect device first
        self.device_type, self.device = detect_device()
        logger.info(f"Using device type: {self.device_type}")

        self._setup_model_and_tokenizer()

    def _get_model_specific_config(self) -> Dict[str, Any]:
        """Get model-specific configuration parameters.
        
        Returns:
            Dict containing model-specific parameters
            
        Raises:
            ModelPreparationError: If model configuration is not found
        """
        try:
            return self.MODEL_CONFIGS[self.model_name]
        except KeyError:
            raise ModelPreparationError(
                f"Model '{self.model_name}' not found in supported configurations. "
                f"Supported models: {list(self.MODEL_CONFIGS.keys())}"
            )

    def _setup_model_and_tokenizer(self) -> None:
        """Initialize model and tokenizer."""
        try:
            # Initialize tokenizer
            logger.info(f"Loading tokenizer for model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                padding_side="right"
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Use our device detection utility
            device_type, _ = detect_device()
            
            # Load model with standard settings
            logger.info(f"Loading model: {self.model_name}")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                device_map=None,
                trust_remote_code=True
            ).to(self.device)

            # Enable gradient checkpointing for memory efficiency
            self.model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
            
            # Debug: Print model layer names
            logger.info("Available model layers:")
            for name, _ in self.model.named_modules():
                if any(keyword in name for keyword in ['attention', 'mlp', 'dense', 'proj']):
                    logger.info(f"  - {name}")

            # Get model-specific configuration
            model_config = self._get_model_specific_config()
            logger.info(f"Using configuration for {self.model_name}: {model_config['description']}")
            
            # Configure LoRA with model-specific target modules
            self.lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
                target_modules=model_config["target_modules"],
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
            
            # Configure PEFT model with task-specific settings
            peft_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
                target_modules=self._get_model_specific_config()["target_modules"],
                inference_mode=False,
                # Add this to preserve model attributes
                modules_to_save=['label_names']
            )
            
            # Prepare PEFT model
            self.model = get_peft_model(self.model, peft_config)
            
            # Explicitly set label names on the PEFT model
            if not hasattr(self.model, 'label_names'):
                self.model.label_names = ['labels']
            
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
        """Prepare and tokenize the dataset with progress bar."""
        try:
            progress = ProgressDisplay()
            dataset = None
            tokenized_dataset = None
            
            # Single progress instance for the entire pipeline
            with progress.task("Loading and processing dataset...") as p:
                # Load dataset
                p.update(description="Loading dataset file...")
                dataset = load_dataset('json', data_files=self.dataset_path)
                self.validate_dataset(dataset)
                
                # Tokenize
                p.update(description="Tokenizing dataset...")
                def tokenize_function(examples: Dict[str, Any]) -> Dict[str, Any]:
                    texts = [
                        self.format_instruction(p, r)
                        for p, r in zip(examples['prompt'], examples['response'])
                    ]
                    return self.tokenizer(
                        texts,
                        truncation=True,
                        max_length=self.config.max_seq_length,
                        padding="max_length",
                        return_tensors=None
                    )

                tokenized_dataset = dataset.map(
                    tokenize_function,
                    batched=True,
                    remove_columns=dataset["train"].column_names,
                    desc=None  # Remove the tqdm description to avoid conflicting messages
                )
                
                # Split dataset
                p.update(description="Splitting dataset...")
                splits = tokenized_dataset["train"].train_test_split(
                    test_size=0.1,
                    shuffle=True,
                    seed=42
                )
                
                # Final status
                p.console.print(
                    f"[green]✓[/green] Dataset ready: "
                    f"{len(splits['train'])} training examples, "
                    f"{len(splits['test'])} test examples"
                )
            
            return splits

        except Exception as e:
            raise DatasetValidationError(f"Dataset preparation failed: {str(e)}")

    def _setup_training_arguments(self, use_tensorboard: bool) -> TrainingArguments:
        """Configure training arguments.
        
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
            eval_strategy="epoch",
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

    def _get_optimized_trainer(self) -> Type[BaseOptimizedTrainer]:
        """Get the appropriate optimized trainer for the current device."""
        if self.device_type == "cuda":
            from .cuda_optimized_trainer import CUDAOptimizedTrainer
            return CUDAOptimizedTrainer
        elif self.device_type == "mps":
            from .mps_optimized_trainer import MPSOptimizedTrainer
            return MPSOptimizedTrainer
        else:
            return Trainer

    def train(self) -> None:
        """Execute the training process."""
        try:
            # Check tensorboard availability
            try:
                import tensorboard
                use_tensorboard = True
            except ImportError:
                logger.debug("Tensorboard not installed. Training will proceed without tensorboard logging.")
                use_tensorboard = False
            
            # Prepare dataset with progress bar
            logger.info("\n1. Loading and Processing Dataset")
            
            # Create a single spinner instance and pass it to the callback
            progress_callback = TrainingProgressCallback()
            
            # Tokenization
            logger.info("Tokenizing dataset...")
            tokenized_dataset = self.prepare_dataset()
            logger.info("Dataset tokenized and processed")

            # Environment setup
            logger.info("Preparing training environment...")
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )

            # Initialize trainer
            trainer_class = self._get_optimized_trainer()
            trainer = trainer_class(
                model=self.model,
                args=self._setup_training_arguments(use_tensorboard),
                train_dataset=tokenized_dataset["train"],
                eval_dataset=tokenized_dataset["test"],
                data_collator=data_collator,
                callbacks=[progress_callback]
            )
            logger.info("Training environment ready")

            # Start training
            logger.info("\n2. Starting Training Process")
            trainer.train()
            
            # Save adapter
            logger.info("\n3. Saving LoRA Adapter")
            final_checkpoint_path = os.path.join(self.output_dir, "final_adapter")
            self.model.save_pretrained(final_checkpoint_path, safe_serialization=True)
            
            # Clean up and report
            for item in os.listdir(self.output_dir):
                if item.startswith("checkpoint-"):
                    shutil.rmtree(os.path.join(self.output_dir, item))
            
            adapter_size = sum(
                os.path.getsize(os.path.join(final_checkpoint_path, f))
                for f in os.listdir(final_checkpoint_path)
            )
            logger.info(f"Adapter saved (Size: {adapter_size / (1024*1024):.2f} MB)")

        except Exception as e:
            logger.error(f"Error: {str(e)}")
            raise