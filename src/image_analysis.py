"""Interactive REPL for Ollama vision model analysis."""

import os
import sys
import base64
import requests
from typing import Optional, Dict, Any, Union
from termcolor import colored
from PIL import Image
import argparse

class OllamaVisionAnalyzer:
    """Class for interacting with Ollama vision model."""
    
    def __init__(
        self,
        model_name: str = "llama3.2-vision:latest",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7
    ):
        """Initialize the vision analyzer.
        
        Args:
            model_name: Name of the Ollama model to use
            base_url: Base URL for Ollama API
            temperature: Generation temperature (0.0 to 1.0)
        """
        self.model_name = model_name
        self.base_url = base_url.rstrip('/')
        self.temperature = temperature
        self.api_endpoint = f"{self.base_url}/api/generate"
        
        # Verify Ollama connection
        self._check_connection()

    def _check_connection(self) -> None:
        """Verify connection to Ollama server.
        
        Raises:
            ConnectionError: If cannot connect to Ollama
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code != 200:
                raise ConnectionError(
                    f"Ollama server returned status code {response.status_code}"
                )
            print(f"Successfully connected to Ollama at {self.base_url}")
            
        except requests.exceptions.RequestException as e:
            raise ConnectionError(
                f"Failed to connect to Ollama server at {self.base_url}: {str(e)}"
            )

    def _encode_image(self, image_path: str) -> str:
        """Encode image as base64 string.
        
        Args:
            image_path: Path to image file
            
        Returns:
            str: Base64 encoded image
            
        Raises:
            FileNotFoundError: If image file not found
            ValueError: If file is not a valid image
        """
        try:
            # Verify it's a valid image
            img = Image.open(image_path)
            img.verify()
            
            # Read and encode
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
                
        except FileNotFoundError:
            raise FileNotFoundError(f"Image file not found: {image_path}")
        except Exception as e:
            raise ValueError(f"Invalid image file: {str(e)}")

    def generate_response(
        self,
        prompt: str,
        image_path: Optional[str] = None
    ) -> str:
        """Generate response from the model.
        
        Args:
            prompt: Text prompt
            image_path: Optional path to image file
            
        Returns:
            str: Model's response
            
        Raises:
            requests.exceptions.RequestException: If API call fails
        """
        # Prepare request payload
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": self.temperature,
            "stream": False
        }
        
        # Add image if provided
        if image_path:
            try:
                base64_image = self._encode_image(image_path)
                payload["images"] = [base64_image]
            except (FileNotFoundError, ValueError) as e:
                print(colored(f"Error with image: {str(e)}", 'red'))
                return "Error: Failed to process image"
        
        try:
            print(f"Sending request to: {self.api_endpoint}")  # Debug line
            print(f"With model: {self.model_name}")           # Debug line
            response = requests.post(self.api_endpoint, json=payload)
            response.raise_for_status()
            return response.json()['response']
            
        except requests.exceptions.RequestException as e:
            error_msg = f"API request failed: {str(e)}"
            print(colored(error_msg, 'red'))
            return f"Error: {error_msg}"

    def repl(self):
        """Start interactive REPL session."""
        print("\nEntering interactive mode with Ollama Vision Model")
        print("Commands:")
        print("  help          - Show this help message")
        print("  quit/exit     - Exit the session")
        print("  temp X        - Set temperature (e.g., temp 0.8)")
        print("  image PATH    - Analyze an image")
        print("  clear         - Clear any loaded image")
        
        current_image: Optional[str] = None
        
        while True:
            try:
                # Show prompt with image status
                image_status = f"[Image: {os.path.basename(current_image)}]" if current_image else ""
                user_input = input(f"\n{image_status}> ").strip()
                
                # Handle commands
                if user_input.lower() in ['quit', 'exit']:
                    print("Goodbye!")
                    break
                    
                elif user_input.lower() == 'help':
                    print("\nCommands:")
                    print("  help          - Show this help message")
                    print("  quit/exit     - Exit the session")
                    print("  temp X        - Set temperature (e.g., temp 0.8)")
                    print("  image PATH    - Analyze an image")
                    print("  clear         - Clear any loaded image")
                    continue
                    
                elif user_input.lower().startswith('temp '):
                    try:
                        new_temp = float(user_input.split()[1])
                        if 0 <= new_temp <= 1:
                            self.temperature = new_temp
                            print(f"Temperature set to {new_temp}")
                        else:
                            print("Temperature must be between 0 and 1")
                    except (IndexError, ValueError):
                        print("Invalid temperature value")
                    continue
                    
                elif user_input.lower().startswith('image '):
                    image_path = user_input[6:].strip()
                    try:
                        # Verify image is valid
                        self._encode_image(image_path)
                        current_image = image_path
                        print(f"Image loaded: {os.path.basename(image_path)}")
                    except (FileNotFoundError, ValueError) as e:
                        print(colored(f"Error: {str(e)}", 'red'))
                    continue
                    
                elif user_input.lower() == 'clear':
                    current_image = None
                    print("Image cleared")
                    continue
                    
                elif not user_input:
                    continue
                
                # Generate and display response
                print("\nThinking...", end='\r')
                response = self.generate_response(user_input, current_image)
                print(" " * 20, end='\r')  # Clear "Thinking..."
                print(colored(response, 'blue'))
                
            except KeyboardInterrupt:
                print("\nInterrupted. Type 'quit' to exit.")
            except Exception as e:
                print(colored(f"Error: {str(e)}", 'red'))

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Ollama Vision Model Interface")
    parser.add_argument(
        "--model",
        type=str,
        default="llama3.2-vision:latest",
        help="Name of the Ollama model"
    )
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:11434",
        help="Ollama API base URL"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Generation temperature (0-1)"
    )
    
    args = parser.parse_args()
    
    try:
        analyzer = OllamaVisionAnalyzer(
            model_name=args.model,
            base_url=args.url,
            temperature=args.temperature
        )
        analyzer.repl()
    except ConnectionError as e:
        print(colored(f"Failed to connect to Ollama: {str(e)}", 'red'))
        sys.exit(1)
    except Exception as e:
        print(colored(f"Unexpected error: {str(e)}", 'red'))
        sys.exit(1)

if __name__ == "__main__":
    main()
