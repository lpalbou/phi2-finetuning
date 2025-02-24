"""Rich-based progress display utilities."""

from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn
)
from rich.console import Console
from typing import Optional
import contextlib

class ProgressDisplay:
    """Progress display utility using Rich."""
    
    def __init__(self):
        """Initialize the progress display."""
        self.console = Console()
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(complete_style="green"),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=self.console,
            expand=True
        )
        self.current_task_id = None
        
    def start(self, description: str, total: Optional[int] = None) -> None:
        """Start a new progress display.
        
        Args:
            description: Task description
            total: Optional total steps (for progress bar)
        """
        self.progress.start()
        self.current_task_id = self.progress.add_task(description, total=total)
    
    def update(self, description: Optional[str] = None, advance: int = 0) -> None:
        """Update the progress display.
        
        Args:
            description: New description (if None, keeps current)
            advance: Number of steps to advance
        """
        if self.current_task_id is not None:
            if description:
                self.progress.update(self.current_task_id, description=description)
            if advance:
                self.progress.advance(self.current_task_id, advance)
                
    def stop(self, final_message: Optional[str] = None) -> None:
        """Stop the progress display.
        
        Args:
            final_message: Optional final message to display
        """
        self.progress.stop()
        if final_message:
            self.console.print(f"[green]✓[/green] {final_message}")

    @contextlib.contextmanager
    def task(self, description: str, total: Optional[int] = None):
        """Context manager for progress tasks.
        
        Args:
            description: Task description
            total: Optional total steps
        """
        try:
            self.start(description, total)
            yield self
        finally:
            self.stop() 