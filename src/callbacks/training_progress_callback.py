"""Training progress callback module using Rich progress bars."""

import logging
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
from transformers import TrainerCallback
from typing import Dict, Optional
import torch
from datetime import datetime

# Configure logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainingProgressCallback(TrainerCallback):
    """Rich-based training progress callback with enhanced display."""
    
    def __init__(self):
        """Initialize the callback with Rich components."""
        super().__init__()
        self.console = Console()
        self.start_time = datetime.now()
        self._trainer = None
        self.current_epoch = 0
        self.total_epochs = 0
        
        # Single progress instance
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=50),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=self.console,
            expand=True
        )

    def _format_memory(self, value_in_bytes: float) -> str:
        """Format memory values in human-readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if value_in_bytes < 1024:
                return f"{value_in_bytes:.2f}{unit}"
            value_in_bytes /= 1024
        return f"{value_in_bytes:.2f}TB"

    def _get_memory_info(self) -> str:
        """Get formatted memory usage string."""
        try:
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated()
                reserved = torch.cuda.memory_reserved()
                return f"GPU: {self._format_memory(allocated)} (Reserved: {self._format_memory(reserved)})"
            elif torch.backends.mps.is_available():
                allocated = torch.mps.current_allocated_memory()
                return f"MPS: {self._format_memory(allocated)}"
            return "CPU"
        except Exception:
            return "N/A"

    def _print_summary_table(self, state, args):
        """Print the training summary table."""
        table = Table(show_header=True, header_style="bold magenta", title="Training Configuration")
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Examples", str(len(state.train_dataloader)))
        table.add_row("Epochs", str(args.num_train_epochs))
        table.add_row("Batch Size", str(args.per_device_train_batch_size))
        table.add_row("Total Batch Size", str(args.per_device_train_batch_size * args.gradient_accumulation_steps))
        table.add_row("Gradient Accumulation", str(args.gradient_accumulation_steps))
        table.add_row("Optimization Steps", str(state.max_steps))
        table.add_row("Memory Usage", self._get_memory_info())
        
        self.console.print("\n")
        self.console.print(Panel(table, border_style="blue"))
        self.console.print("\n")

    def on_train_begin(self, args, state, control, **kwargs):
        """Initialize and display training progress."""
        self._trainer = kwargs.get('trainer')
        self.total_epochs = int(args.num_train_epochs)
        
        # Print initial summary
        self._print_summary_table(state, args)
        
        # Start progress tracking
        self.progress.start()
        self.task_id = self.progress.add_task(
            description="[cyan]Training Progress",
            total=state.max_steps
        )

    def on_step_end(self, args, state, control, **kwargs):
        """Update progress on step completion."""
        if state.global_step % args.logging_steps == 0:
            # Update progress
            self.progress.update(
                self.task_id,
                advance=args.logging_steps,
                description=f"[cyan]Epoch {self.current_epoch}/{self.total_epochs}"
            )

    def on_epoch_begin(self, args, state, control, **kwargs):
        """Update display on epoch start."""
        self.current_epoch = int(state.epoch) + 1
        self.progress.update(
            self.task_id,
            description=f"[cyan]Starting Epoch {self.current_epoch}/{self.total_epochs}"
        )

    def on_epoch_end(self, args, state, control, **kwargs):
        """Update display on epoch completion."""
        elapsed_time = datetime.now() - self.start_time
        avg_time_per_epoch = elapsed_time / self.current_epoch if self.current_epoch > 0 else 0
        
        self.console.print(f"\n[green]Epoch {self.current_epoch}/{self.total_epochs} Complete")
        self.console.print(f"Time: {elapsed_time.total_seconds():.1f}s • Avg: {avg_time_per_epoch.total_seconds():.1f}s/epoch")
        self.console.print(f"Memory: {self._get_memory_info()}\n")

    def on_train_end(self, args, state, control, **kwargs):
        """Display final training summary."""
        self.progress.stop()
        
        total_time = datetime.now() - self.start_time
        self.console.print("\n[bold green]Training Complete!")
        self.console.print(f"Total Time: {total_time.total_seconds():.1f}s")
        self.console.print(f"Final Memory: {self._get_memory_info()}")
        
        self.console.print("\n[cyan]To try your model:")
        self.console.print("1. Interactive mode: python -m src.dialogue --adapter_path output/final_adapter")
        self.console.print("2. Compare with base: python compare_models.py\n")