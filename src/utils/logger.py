"""
Logging configuration and utilities.

This module provides centralized logging configuration for the entire
SafetyKnob system with proper formatting and file handling.
"""

import os
import logging
import logging.handlers
from datetime import datetime
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output."""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        # Add color to level name
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.RESET}"
        
        # Format the message
        formatted = super().format(record)
        
        # Reset levelname for other handlers
        record.levelname = levelname
        
        return formatted


def setup_logging(
    name: str = "safetyknob",
    level: str = "INFO",
    log_dir: str = "logs",
    console: bool = True,
    file: bool = True,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        name: Logger name
        level: Logging level
        log_dir: Directory for log files
        console: Whether to log to console
        file: Whether to log to file
        format_string: Custom format string
        
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    logger.handlers = []
    
    # Default format
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, level.upper()))
        
        # Use colored formatter for console
        console_formatter = ColoredFormatter(format_string)
        console_handler.setFormatter(console_formatter)
        
        logger.addHandler(console_handler)
    
    # File handler
    if file:
        os.makedirs(log_dir, exist_ok=True)
        
        # Create log filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'{name}_{timestamp}.log')
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(getattr(logging, level.upper()))
        
        # Regular formatter for file
        file_formatter = logging.Formatter(format_string)
        file_handler.setFormatter(file_formatter)
        
        logger.addHandler(file_handler)
        
        # Also create a symlink to latest log
        latest_link = os.path.join(log_dir, f'{name}_latest.log')
        if os.path.exists(latest_link):
            os.remove(latest_link)
        os.symlink(os.path.basename(log_file), latest_link)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class LoggerContext:
    """Context manager for temporary logging configuration."""
    
    def __init__(self, logger: logging.Logger, level: Optional[str] = None):
        """
        Initialize logger context.
        
        Args:
            logger: Logger instance
            level: Temporary logging level
        """
        self.logger = logger
        self.original_level = logger.level
        self.temp_level = getattr(logging, level.upper()) if level else None
    
    def __enter__(self):
        if self.temp_level is not None:
            self.logger.setLevel(self.temp_level)
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.setLevel(self.original_level)


def log_function_call(logger: logging.Logger):
    """
    Decorator to log function calls and returns.
    
    Args:
        logger: Logger instance
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Log function call
            logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
            
            try:
                # Call function
                result = func(*args, **kwargs)
                
                # Log successful return
                logger.debug(f"{func.__name__} returned successfully")
                
                return result
                
            except Exception as e:
                # Log exception
                logger.error(f"{func.__name__} raised exception: {e}", exc_info=True)
                raise
        
        return wrapper
    return decorator


def log_execution_time(logger: logging.Logger):
    """
    Decorator to log function execution time.
    
    Args:
        logger: Logger instance
    """
    import time
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                logger.info(f"{func.__name__} executed in {execution_time:.2f} seconds")
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"{func.__name__} failed after {execution_time:.2f} seconds")
                raise
        
        return wrapper
    return decorator


# Configure root logger for the package
def configure_root_logger(level: str = "INFO"):
    """Configure the root logger for the entire package."""
    root_logger = logging.getLogger('safetyknob')
    
    # Only configure if not already configured
    if not root_logger.handlers:
        setup_logging(
            name='safetyknob',
            level=level,
            console=True,
            file=True
        )


# Convenience functions
def debug(msg: str, *args, **kwargs):
    """Log debug message."""
    logging.getLogger('safetyknob').debug(msg, *args, **kwargs)


def info(msg: str, *args, **kwargs):
    """Log info message."""
    logging.getLogger('safetyknob').info(msg, *args, **kwargs)


def warning(msg: str, *args, **kwargs):
    """Log warning message."""
    logging.getLogger('safetyknob').warning(msg, *args, **kwargs)


def error(msg: str, *args, **kwargs):
    """Log error message."""
    logging.getLogger('safetyknob').error(msg, *args, **kwargs)


def critical(msg: str, *args, **kwargs):
    """Log critical message."""
    logging.getLogger('safetyknob').critical(msg, *args, **kwargs)