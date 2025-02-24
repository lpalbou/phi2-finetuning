"""Interactive dialogue module for Phi-2 with LoRA weights."""

import os
import sys
from typing import Optional
import torch
from termcolor import colored
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

class ModelDialogue:
    """Class for interactive dialogue with the model."""
    
    def __init__(
        self,
        base_model_name: str = "microsoft/phi-2",
        adapter_path: str = "output/lpa-smaller-unique-gpt/final_adapter",
        max_length: int = 512,
        temperature: float = 0.7,
        device: Optional[str] = None
    ):
        """Initialize the dialogue system.
        
        Args:
            base_model_name: Name of the base model
            adapter_path: Path to the LoRA adapter weights
            max_length: Maximum sequence length for generation
            temperature: Sampling temperature (higher = more random)
            device: Device to run the model on (default: auto-detect)
        """
        self.max_length = max_length
        self.temperature = temperature
        
        # Auto-detect device if not specified
        if device is None:
            self.device = (
                torch.device("mps") 
                if torch.backends.mps.is_available()
                else torch.device("cpu")
            )
        else:
            self.device = torch.device(device)
            
        print(f"Using device: {self.device}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            padding_side="right"
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        print("Loading base model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True
        ).to(self.device)
        
        # Load LoRA adapter if path exists
        if os.path.exists(adapter_path):
            print(f"Loading LoRA adapter from {adapter_path}")
            self.model = PeftModel.from_pretrained(
                self.model,
                adapter_path
            )
        else:
            print(f"Warning: Adapter path {adapter_path} not found. Using base model only.")
            
        self.model.eval()

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
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Interactive dialogue with Phi-2 + LoRA")
    parser.add_argument(
        "--adapter_path",
        type=str,
        default="output/lpa-smaller-unique-gpt/final_adapter",
        help="Path to LoRA adapter weights"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length"
    )
    args = parser.parse_args()
    
    # Create and start dialogue
    dialogue = ModelDialogue(
        adapter_path=args.adapter_path,
        temperature=args.temperature,
        max_length=args.max_length
    )
    dialogue.repl()

if __name__ == "__main__":
    main()
