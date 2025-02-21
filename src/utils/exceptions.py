"""Custom exceptions for the Phi-2 fine-tuning project."""

class DatasetValidationError(Exception):
    """Exception raised when dataset validation fails.
    
    This exception is raised when there are issues with:
    - Dataset structure
    - Missing required fields
    - Invalid data types
    - Invalid data values
    """
    pass


class ModelPreparationError(Exception):
    """Exception raised when model preparation fails.
    
    This exception is raised when there are issues with:
    - Model loading
    - Model configuration
    - LoRA adaptation
    - Device placement
    """
    pass


class TrainingError(Exception):
    """Exception raised when training process fails.
    
    This exception is raised when there are issues with:
    - Training loop
    - Loss computation
    - Gradient updates
    - Resource allocation
    """
    pass


class DeviceError(Exception):
    """Exception raised when there are device-related issues.
    
    This exception is raised when there are issues with:
    - MPS availability
    - Memory allocation
    - Device compatibility
    """
    pass