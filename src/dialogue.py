"""Interactive dialogue module for Phi-2 with LoRA weights."""

import os
import sys
from typing import Optional
import torch
from termcolor import colored
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from utils.device_utils import detect_device

class ModelDialogue:
    """Class for interactive dialogue with the model."""
    
    def __init__(
        self,
        base_model_name: str = "microsoft/phi-2",
        adapter_path: Optional[str] = None,
        max_length: int = 512,
        temperature: float = 0.7,
        device: Optional[str] = None
    ):
        """Initialize the dialogue system.
        
        Args:
            base_model_name: Name of the base model from HuggingFace
            adapter_path: Optional path to LoRA adapter weights
            max_length: Maximum sequence length for generation
            temperature: Sampling temperature (higher = more random)
            device: Device to run the model on (default: auto-detect)
        """
        self.max_length = max_length
        self.temperature = temperature
        
        # Use device_utils to detect the best available device
        device_type, device_str = detect_device()
        self.device = torch.device(device_str if device is None else device)
            
        print(f"Using device: {self.device}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            padding_side="right"
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model with appropriate dtype based on device
        print(f"Loading base model: {base_model_name}...")
        dtype = torch.float16 if device_type == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=dtype,
            trust_remote_code=True
        ).to(self.device)
        
        # Load LoRA adapter if specified and exists
        if adapter_path:
            if os.path.exists(adapter_path):
                print(f"Loading LoRA adapter from {adapter_path}")
                self.model = PeftModel.from_pretrained(
                    self.model,
                    adapter_path
                )
            else:
                print(f"Warning: Adapter path {adapter_path} not found. Using base model only.")
            
        self.model.eval()
        
        # Additional optimization for CUDA
        if device_type == "cuda":
            self.model = self.model.half()  # Convert to FP16 for better CUDA performance

    def generate_response(self, prompt: str) -> str:
        """Generate a response for the given prompt.
        
        Args:
            prompt: Input prompt
            
        Returns:
            str: Generated response
        """
        # Format the prompt
        formatted_prompt = (
            "Below is an instruction that describes a task. "
            "Write a response that completes the request.\n\n"
            "### Instruction:\n"
            f"{prompt}\n\n"
            "### Response:\n"
        )
        
        # Tokenize
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=self.max_length,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
        # Decode and clean up response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.replace(formatted_prompt, "").strip()
        return response

    def repl(self):
        """Start an interactive REPL session."""
        print("\nEntering interactive mode. Type 'quit' or 'exit' to end the session.")
        print("Type 'help' for commands.")
        
        while True:
            try:
                # Get input
                user_input = input("\n> ").strip()
                
                # Handle commands
                if user_input.lower() in ['quit', 'exit']:
                    print("Goodbye!")
                    break
                elif user_input.lower() == 'help':
                    print("\nCommands:")
                    print("  help  - Show this help message")
                    print("  quit  - Exit the session")
                    print("  exit  - Exit the session")
                    print("  temp X  - Set temperature to X (e.g., temp 0.8)")
                    continue
                elif user_input.lower().startswith('temp '):
                    try:
                        new_temp = float(user_input.split()[1])
                        self.temperature = new_temp
                        print(f"Temperature set to {new_temp}")
                    except (IndexError, ValueError):
                        print("Invalid temperature value")
                    continue
                elif not user_input:
                    continue
                
                # Generate and display response
                print("\nThinking...", end='\r')
                response = self.generate_response(user_input)
                print(" " * 20, end='\r')  # Clear "Thinking..."
                print(colored(response, 'blue'))
                
            except KeyboardInterrupt:
                print("\nInterrupted. Type 'quit' to exit.")
            except Exception as e:
                print(f"Error: {str(e)}")

def main():
    """Main entry point."""
    import argparse
    
    # Create parser with more detailed help
    parser = argparse.ArgumentParser(
        description="Interactive dialogue with Large Language Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Will show the help message
  python dialogue.py
  
  # Use Phi-2 with a LoRA adapter
  python dialogue.py --adapter_path output/my_adapter/final_adapter
  
  # Use a different model with custom parameters
  python dialogue.py --base_model_name meta-llama/Llama-2-7b-chat-hf --temperature 0.8
  
  # Use specific device
  python dialogue.py --device cuda:0
        """
    )
    
    parser.add_argument(
        "--base_model_name",
        type=str,
        default="microsoft/phi-2",
        help="HuggingFace model to use (default: microsoft/phi-2)"
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        help="Optional path to LoRA adapter weights"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature - higher values make output more random (default: 0.7)"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length for generation (default: 512)"
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Device to run on (default: auto-detect best available)"
    )
    
    # If no arguments provided, print help and exit
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
    
    args = parser.parse_args()
    
    # Create and start dialogue
    dialogue = ModelDialogue(
        base_model_name=args.base_model_name,
        adapter_path=args.adapter_path,
        temperature=args.temperature,
        max_length=args.max_length,
        device=args.device
    )
    dialogue.repl()

if __name__ == "__main__":
    main()
