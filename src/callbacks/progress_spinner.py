"""Progress spinner utility for displaying indeterminate progress."""

import sys
import threading
import time
from typing import Optional

class ProgressSpinner:
    """Simple spinner animation for indeterminate progress."""
    def __init__(self, message: str):
        self.message = message
        self.frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self.current = 0
        self.active = True
        self._spin_thread = None
        
    def start(self):
        """Start the spinner animation in a separate thread."""
        self.active = True
        self._spin_thread = threading.Thread(target=self._spin_loop)
        self._spin_thread.daemon = True
        self._spin_thread.start()
    
    def _spin_loop(self):
        """Animation loop running in separate thread."""
        while self.active:
            frame = self.frames[self.current]
            sys.stdout.write(f'\r{frame} {self.message}')
            sys.stdout.flush()
            self.current = (self.current + 1) % len(self.frames)
            time.sleep(0.1)  # Control animation speed
            
    def stop(self, final_message: Optional[str] = None):
        """Stop spinner and show final message."""
        self.active = False
        if self._spin_thread:
            self._spin_thread.join()
        if final_message:
            sys.stdout.write(f'\r✓ {final_message}\n')
        else:
            sys.stdout.write('\r✓ Done!\n')
        sys.stdout.flush() 