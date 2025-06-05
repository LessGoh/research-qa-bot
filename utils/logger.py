"""
Logging configuration for Research Q&A Bot
"""
import logging
import logging.handlers
from pathlib import Path
from typing import Optional
from .config import config


def setup_logger(
    name: str = "research_qa_bot",
    log_file: Optional[str] = None,
    log_level: Optional[str] = None,
    max_size_mb: int = 10,
    backup_count: int = 5
) -> logging.Logger:
    """
    Setup logger with file and console handlers
    
    Args:
        name: Logger name
        log_file: Path to log file (defaults to config)
        log_level: Logging level (defaults to config)
        max_size_mb: Maximum log file size in MB
        backup_count: Number of backup files to keep
        
    Returns:
        Configured logger
    """
    # Get configuration values
    log_file = log_file or config.log_file
    log_level = log_level or config.log_level
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        config.get("logging.format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler with rotation
    if log_file:
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_size_mb * 1024 * 1024,  # Convert MB to bytes
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "research_qa_bot") -> logging.Logger:
    """
    Get existing logger or create new one
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    
    # If logger has no handlers, set it up
    if not logger.handlers:
        return setup_logger(name)
    
    return logger


class LoggerMixin:
    """Mixin class to add logging capability to other classes"""
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class"""
        return get_logger(f"research_qa_bot.{self.__class__.__name__}")


def log_function_call(func):
    """Decorator to log function calls"""
    def wrapper(*args, **kwargs):
        logger = get_logger("research_qa_bot.function_calls")
        logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        
        try:
            result = func(*args, **kwargs)
            logger.debug(f"{func.__name__} completed successfully")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} failed with error: {e}")
            raise
    
    return wrapper


def log_query(query: str, response_time: float, mode: str = "unknown"):
    """
    Log user queries for analytics
    
    Args:
        query: User query text
        response_time: Time taken to process query
        mode: Research mode used
    """
    query_logger = get_logger("research_qa_bot.queries")
    query_logger.info(
        f"Query processed | Mode: {mode} | Time: {response_time:.2f}s | "
        f"Query: {query[:100]}{'...' if len(query) > 100 else ''}"
    )


def log_error(error: Exception, context: str = ""):
    """
    Log errors with context
    
    Args:
        error: Exception that occurred
        context: Additional context information
    """
    error_logger = get_logger("research_qa_bot.errors")
    error_logger.error(f"Error in {context}: {str(error)}", exc_info=True)


# Initialize default logger
default_logger = setup_logger()
