"""Compare responses between original Phi-2 and LoRA fine-tuned version."""

import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from typing import List, Dict
from termcolor import colored

class ModelComparison:
    def __init__(self, base_model_name: str, adapter_path: str):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Initialize tokenizer
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=True
        )
        
        # Load base model
        print("Loading base model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True
        ).to(self.device)
        
        self.adapter_path = adapter_path
        self.test_questions = [
            "What is quantum computing?",
            "Explain how a car engine works",
            "What is the theory of relativity?",
            "How does photosynthesis work?",
            "What is artificial intelligence?"
        ]
        
    def get_base_response(self, prompt: str) -> str:
        """Get response from base model."""
        instruction = f"""Below is an instruction that describes a task.

### Instruction:
{prompt}

### Response:
"""
        inputs = self.tokenizer(instruction, return_tensors="pt").to(self.device)
        
        outputs = self.model.generate(
            **inputs,
            max_length=512,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def apply_lora(self):
        """Apply LoRA adapter to base model."""
        print("\nApplying LoRA adapter...")
        self.model = PeftModel.from_pretrained(self.model, self.adapter_path)
        print("LoRA adapter applied successfully!")
        
    def get_lora_response(self, prompt: str) -> str:
        """Get response from LoRA-enhanced model."""
        instruction = f"""Below is an instruction that describes a task. Write a response that completes the request in a humorous and engaging way.

### Instruction:
{prompt}

### Response:
"""
        inputs = self.tokenizer(instruction, return_tensors="pt").to(self.device)
        
        outputs = self.model.generate(
            **inputs,
            max_length=512,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def compare_responses(self):
        """Compare responses from both models."""
        print("\nGenerating responses from base model...")
        base_responses = {}
        for question in self.test_questions:
            base_responses[question] = self.get_base_response(question)
            
        # Apply LoRA adapter
        self.apply_lora()
        
        print("\nGenerating responses from fine-tuned model...")
        lora_responses = {}
        for question in self.test_questions:
            lora_responses[question] = self.get_lora_response(question)
            
        # Print comparisons
        for question in self.test_questions:
            print("\n" + "="*100)
            print(f"Question: {question}")
            print("-"*100)
            
            # Print original response in blue
            print(colored("Original Model Response:", "blue", attrs=["bold"]))
            print(colored(base_responses[question].replace(
                "Below is an instruction that describes a task.", ""
            ).strip(), "blue"))
            
            print("-"*100)
            
            # Print fine-tuned response in green
            print(colored("Fine-tuned Model Response:", "green", attrs=["bold"]))
            print(colored(lora_responses[question].replace(
                "Below is an instruction that describes a task. Write a response that completes the request in a humorous and engaging way.", ""
            ).strip(), "green"))
            
            print("="*100)
            input("\nPress Enter for next comparison...")

def main():
    base_model = "microsoft/phi-2"
    adapter_path = "./output/final_adapter"
    
    try:
        comparison = ModelComparison(base_model, adapter_path)
        comparison.compare_responses()
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
    finally:
        # Clean up
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()

if __name__ == "__main__":
    main()