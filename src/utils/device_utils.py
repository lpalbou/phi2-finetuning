"""Device detection and configuration utilities."""

import logging
import torch
from typing import Tuple, Literal

logger = logging.getLogger(__name__)

DeviceType = Literal["cuda", "mps", "cpu"]

def detect_device() -> Tuple[DeviceType, str]:
    """Detect and return the best available device.
    
    Returns:
        Tuple[DeviceType, str]: Device type and device string
        
    Example:
        device_type, device_str = detect_device()
        # Returns: ("cuda", "cuda:0") or ("mps", "mps") or ("cpu", "cpu")
    """
    if torch.cuda.is_available():
        device_type = "cuda"
        device_str = f"cuda:{torch.cuda.current_device()}"
        return device_type, device_str
        
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device_type = "mps"
        device_str = "mps"
        return device_type, device_str
        
    device_type = "cpu"
    device_str = "cpu"
    logger.warning("No GPU detected, using CPU")
    return device_type, device_str 