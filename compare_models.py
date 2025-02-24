"""Compare responses between original Phi-2 and LoRA fine-tuned version."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from termcolor import colored
import json
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
        
    def _load_base_model(self):
        """Load the base model and tokenizer."""
        print("\nLoading base model...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float32,
            device_map=self.device
        )
        
    def _apply_lora_adapter(self):
        """Apply LoRA adapter to the existing model."""
        print("\nApplying LoRA adapter...")
        self.model = PeftModel.from_pretrained(self.model, self.adapter_path)

    def _cleanup_model(self):
        """Clean up model from memory."""
        del self.model
        del self.tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("\nCleared model from memory")

    def generate_response(self, question: str) -> str:
        """Generate response for a single question."""
        # Simple direct prompt without Q&A formatting
        prompt = f"{self.system_prompt}\n\n{question}"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        outputs = self.model.generate(
            **inputs,
            **self.generation_config,
            pad_token_id=self.tokenizer.pad_token_id
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the system prompt from the response if it appears
        response = response.replace(self.system_prompt, "").strip()
        return response

    def compare_responses(self, questions):
        """Compare responses from both models."""
        # Load model once
        self._load_base_model()
        
        # Get base responses
        print("\nGenerating base model responses...")
        base_responses = {q: self.generate_response(q) for q in questions}
        
        # Apply LoRA adapter to same model
        self._apply_lora_adapter()
        
        # Get LoRA responses
        print("\nGenerating LoRA model responses...")
        lora_responses = {q: self.generate_response(q) for q in questions}
        
        # Display comparisons
        for question in questions:
            print(f"\nquestion: {colored(question, 'red')}")
            print(f"base model: {colored(base_responses[question], 'blue')}")
            print(f"altered model: {colored(lora_responses[question], 'green')}")
            print("\n" + "="*80)
            input("\nPress Enter for next comparison...")
        
        # Clean up
        self._cleanup_model()

    def generate_base_responses(self, questions):
        """Generate and save responses from base model."""
        self._load_base_model()
        base_responses = {}
        
        print("\nGenerating base model responses...")
        for question in questions:
            response = self.generate_response(question)
            base_responses[question] = response
            
        self._cleanup_model()
        return base_responses

    def generate_lora_responses(self, questions):
        """Generate responses from LoRA-enhanced model."""
        self._load_base_model()
        lora_responses = {}
        
        print("\nGenerating LoRA model responses...")
        for question in questions:
            response = self.generate_response(question)
            lora_responses[question] = response
            
        self._cleanup_model()
        return lora_responses

    def verify_adapter_effect(self, test_prompt="Test question to verify adapter effect."):
        """Verify that the adapter is actually changing model outputs."""
        print("\nVerifying adapter effect...")
        
        # Generate with base model
        base_output = self.generate_response(test_prompt)
        print(f"\nBase model output: {base_output[:100]}...")
        
        # Apply adapter
        self._apply_lora_adapter()
        
        # Generate with adapted model
        adapted_output = self.generate_response(test_prompt)
        print(f"\nAdapted model output: {adapted_output[:100]}...")
        
        # Verify outputs are different
        is_different = base_output != adapted_output
        print(f"\nOutputs are different: {is_different}")
        
        return is_different

def main():
    base_model = "microsoft/phi-2"
    adapter_path = "./output/final_adapter"
    
    try:
        comparison = ModelComparator(base_model, adapter_path)
        questions = [
            "How does the MIIC-SDG study demonstrates the importance of synthetic data in preserving confidentiality while maintaining useful insights for analysis?",
            "What new methods has MIIC-SDG introduced for evaluating the identifiability of synthetic data (MIS) and assessing the trade-off between quality and privacy (Quality-Privacy scores)?",
            "What implications does MIIC-SDG has for future directions in digital medicine and other related fields utilizing synthetic data for privacy preservation?"
        ]
        comparison.compare_responses(questions)
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