"""Logging utilities with timestamps."""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Union, Optional


def setup_logger(
    name: str,
    log_file: Optional[Union[str, Path]] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Set up a logger with timestamps and optional file output.
    
    Args:
        name: Logger name
        log_file: Optional path to log file
        level: Logging level (default: INFO)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear any existing handlers
    logger.handlers = []
    
    # Create formatter with timestamps
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def log_progress(logger: logging.Logger, current: int, total: int, prefix: str = "") -> None:
    """
    Log progress with percentage.
    
    Args:
        logger: Logger instance
        current: Current item number
        total: Total number of items
        prefix: Optional prefix message
    """
    percentage = (current / total) * 100 if total > 0 else 0
    msg = f"{prefix}Progress: {current}/{total} ({percentage:.1f}%)"
    logger.info(msg)
