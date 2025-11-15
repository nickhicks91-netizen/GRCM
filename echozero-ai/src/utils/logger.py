"""
Logging utilities
"""
import logging
import sys
from pathlib import Path
from typing import Optional
from torch.utils.tensorboard import SummaryWriter


class Logger:
    """
    Combined console and TensorBoard logger
    """

    def __init__(
        self,
        log_dir: str,
        name: str = 'echozero',
        use_tensorboard: bool = True,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Console logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        # Clear existing handlers
        self.logger.handlers = []

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # File handler
        file_handler = logging.FileHandler(self.log_dir / 'training.log')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # TensorBoard writer
        self.writer = None
        if use_tensorboard:
            self.writer = SummaryWriter(log_dir=str(self.log_dir / 'tensorboard'))

    def info(self, message: str):
        """Log info message"""
        self.logger.info(message)

    def warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)

    def error(self, message: str):
        """Log error message"""
        self.logger.error(message)

    def log_scalar(self, tag: str, value: float, step: int):
        """Log scalar to TensorBoard"""
        if self.writer is not None:
            self.writer.add_scalar(tag, value, step)

    def log_scalars(self, tag: str, values: dict, step: int):
        """Log multiple scalars to TensorBoard"""
        if self.writer is not None:
            self.writer.add_scalars(tag, values, step)

    def close(self):
        """Close logger"""
        if self.writer is not None:
            self.writer.close()
