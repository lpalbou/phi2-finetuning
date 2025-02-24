"""Compare responses between original Phi-2 and LoRA fine-tuned version."""

import sys
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from termcolor import colored
import yaml
import gc

class ModelComparator:
    def __init__(self, base_model_name: str, adapter_path: str):
        self.base_model_name = base_model_name
        self.adapter_path = adapter_path
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Set system prompt
        self.system_prompt = "You are here to provide concise answers to questions."
        
        # Set generation parameters
        self.generation_config = {
            "max_length": 512,
            "num_return_sequences": 1,
            "temperature": 0.7,
            "do_sample": True
        }
        
        # Load models once at initialization
        self._initialize_models()
        
    def _initialize_models(self):
        """Load models and tokenizer once at startup."""
        print("\nInitializing models...")
        
        # Load tokenizer
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        
        # Load base model
        print("Loading base model...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float32,
            device_map=self.device
        )
        self.base_model.eval()
        
        # Create LoRA model as a separate instance
        print("Loading LoRA model...")
        self.lora_model = PeftModel.from_pretrained(
            self.base_model,
            self.adapter_path
        )
        self.lora_model.eval()
        
        print("Models ready!")

    def generate_response(self, question: str, use_lora: bool = False) -> str:
        """Generate response using specified model."""
        model = self.lora_model if use_lora else self.base_model
        
        # Use the same format as dialogue.py
        formatted_prompt = (
            "Instruct: {prompt}.\nOutput: "
        ).format(prompt=question)
        
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                **self.generation_config,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the prompt from the response
        response = response.replace(formatted_prompt, "").strip()
        
        return response

    def repl(self):
        """Interactive REPL mode for comparing responses."""
        print("\nEntering interactive mode. Type 'quit' or 'exit' to end the session.")
        print("Type 'help' for commands.")
        
        while True:
            try:
                user_input = input("\n> ").strip()
                
                if user_input.lower() in ['quit', 'exit']:
                    break
                elif user_input.lower() == 'help':
                    print("\nCommands:")
                    print("  help  - Show this help message")
                    print("  quit  - Exit the session")
                    print("  exit  - Exit the session")
                    continue
                elif not user_input:
                    continue
                
                # Generate responses from both models
                base_response = self.generate_response(user_input, use_lora=False)
                lora_response = self.generate_response(user_input, use_lora=True)
                
                # Display results
                print(f"\nQuestion: {colored(user_input, 'red')}")
                print(f"Base model: {colored(base_response, 'blue')}")
                print(f"LoRA model: {colored(lora_response, 'green')}")
                
            except KeyboardInterrupt:
                print("\nInterrupted. Type 'quit' to exit.")
            except Exception as e:
                print(f"Error: {str(e)}")

    def cleanup(self):
        """Clean up resources."""
        del self.base_model
        del self.lora_model
        del self.tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Compare responses between base model and LoRA-adapted version",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Description:
-----------
This tool allows you to compare responses between the original base model and 
its LoRA-adapted version. It helps evaluate the effect of fine-tuning by showing
side-by-side responses from both models.

Usage Modes:
-----------
1. Interactive Mode (default):
   Run without --questions_file to enter REPL mode where you can type questions
   and see responses from both models side by side.

2. Batch Mode:
   Provide a YAML file with questions to process them all at once.

Questions File Format (YAML):
--------------------------
questions:
  - "What is the capital of France?"
  - "Explain quantum computing in simple terms"
  - "Write a haiku about programming"

Examples:
--------
1. Interactive mode with default model (Phi-2):
   python compare_models.py \\
          --adapter_path    output/my_adapter/final_adapter

2. Compare with a different base model:
   python compare_models.py \\
          --base_model     microsoft/phi-2 \\
          --adapter_path   output/my_adapter/final_adapter

3. Batch mode with custom questions:
   python compare_models.py \\
          --adapter_path   output/my_adapter/final_adapter \\
          --questions_file my_test_questions.yaml
        """
    )
    
    parser.add_argument(
        "--base_model",
        type=str,
        default="microsoft/phi-2",
        help="HuggingFace model to use (default: microsoft/phi-2)"
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        required=True,
        help="Path to LoRA adapter weights (required)"
    )
    parser.add_argument(
        "--questions_file",
        type=str,
        help="Optional YAML file containing test questions"
    )
    
    # If no arguments provided, print help and exit
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
    
    args = parser.parse_args()
    
    try:
        comparison = ModelComparator(args.base_model, args.adapter_path)
        
        if args.questions_file:
            # Batch mode with questions file
            try:
                with open(args.questions_file, 'r') as f:
                    yaml_content = yaml.safe_load(f)
                    if not isinstance(yaml_content, dict) or 'questions' not in yaml_content:
                        raise ValueError("YAML file must contain a 'questions' list")
                    questions = yaml_content['questions']
                    if not isinstance(questions, list):
                        raise ValueError("'questions' must be a list of strings")
            except yaml.YAMLError as e:
                print(f"Error reading YAML file: {e}")
                sys.exit(1)
            except Exception as e:
                print(f"Error processing questions file: {e}")
                sys.exit(1)
                
            # Process questions in batch mode
            for question in questions:
                print(f"\nQuestion: {colored(question, 'red')}")
                base_response = comparison.generate_response(question, use_lora=False)
                lora_response = comparison.generate_response(question, use_lora=True)
                print(f"Base model: {colored(base_response, 'blue')}")
                print(f"LoRA model: {colored(lora_response, 'green')}")
                print("\n" + "="*80)
                input("Press Enter for next comparison...")
        else:
            # Interactive REPL mode
            comparison.repl()
            
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
    finally:
        comparison.cleanup()

if __name__ == "__main__":
    main()