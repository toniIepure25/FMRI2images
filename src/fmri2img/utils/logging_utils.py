"""
Logging Utilities
=================

Professional logging setup with consistent formatting, file handlers, and log levels.

Features:
- Hierarchical loggers (per-module)
- Colored console output
- File logging with rotation
- Configurable log levels
- Contextual logging (with timing, memory, etc.)
- Integration with tqdm progress bars

Usage:
    # Simple setup
    from fmri2img.utils.logging_utils import setup_logging
    logger = setup_logging("train_mlp", level="INFO")
    
    # With file output
    logger = setup_logging(
        "train_mlp",
        level="INFO",
        log_file="outputs/logs/train.log"
    )
    
    # With timing context
    from fmri2img.utils.logging_utils import log_time
    
    with log_time(logger, "Training epoch 1"):
        train_epoch(...)
    
    # Progress logging
    from fmri2img.utils.logging_utils import TqdmLoggingHandler
    
    for batch in tqdm(dataloader):
        logger.info(f"Processing batch {i}")  # Won't break progress bar
"""

import logging
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any
from contextlib import contextmanager
from datetime import datetime

# Try to import colorama for colored output
try:
    from colorama import init, Fore, Style
    init(autoreset=True)
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False


class ColoredFormatter(logging.Formatter):
    """
    Colored log formatter for console output.
    
    Log levels are color-coded for better readability:
    - DEBUG: Cyan
    - INFO: Green
    - WARNING: Yellow
    - ERROR: Red
    - CRITICAL: Bold Red
    """
    
    # Color mapping for log levels
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[1;31m', # Bold Red
    }
    RESET = '\033[0m'
    
    def format(self, record):
        """Format log record with colors"""
        if not COLORAMA_AVAILABLE:
            return super().format(record)
        
        # Add color to level name
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.RESET}"
        
        # Format message
        formatted = super().format(record)
        
        # Reset levelname for subsequent handlers
        record.levelname = levelname
        
        return formatted


class TqdmLoggingHandler(logging.Handler):
    """
    Logging handler that works with tqdm progress bars.
    
    Ensures log messages don't interfere with progress bar display.
    """
    
    def emit(self, record):
        """Emit log record using tqdm.write"""
        try:
            from tqdm import tqdm
            msg = self.format(record)
            tqdm.write(msg)
        except ImportError:
            # Fallback to standard output if tqdm not available
            print(self.format(record))
        except Exception:
            self.handleError(record)


def setup_logging(
    name: str,
    level: str = "INFO",
    log_file: Optional[Path] = None,
    console: bool = True,
    format_string: Optional[str] = None,
    date_format: Optional[str] = None,
    use_colors: bool = True,
    use_tqdm: bool = False
) -> logging.Logger:
    """
    Setup logger with consistent formatting and handlers.
    
    Args:
        name: Logger name (typically module name or script name)
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for logging to file
        console: Whether to log to console (default: True)
        format_string: Custom format string for log messages
        date_format: Custom date format for timestamps
        use_colors: Whether to use colored output (default: True)
        use_tqdm: Whether to use tqdm-compatible handler (default: False)
    
    Returns:
        Configured logger instance
    
    Example:
        >>> logger = setup_logging("train_mlp", level="INFO", log_file="train.log")
        >>> logger.info("Starting training")
        2024-12-06 10:30:45 - train_mlp - INFO - Starting training
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Default format strings
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    if date_format is None:
        date_format = "%Y-%m-%d %H:%M:%S"
    
    # Console handler
    if console:
        if use_tqdm:
            console_handler = TqdmLoggingHandler()
        else:
            console_handler = logging.StreamHandler(sys.stdout)
        
        if use_colors and COLORAMA_AVAILABLE:
            console_formatter = ColoredFormatter(format_string, datefmt=date_format)
        else:
            console_formatter = logging.Formatter(format_string, datefmt=date_format)
        
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, mode='a')
        file_formatter = logging.Formatter(format_string, datefmt=date_format)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


def silence_library_loggers(libraries: Optional[list] = None) -> None:
    """
    Silence noisy library loggers.
    
    Args:
        libraries: List of library names to silence (default: common noisy libraries)
    
    Example:
        >>> silence_library_loggers(["nibabel", "urllib3", "PIL"])
    """
    if libraries is None:
        libraries = [
            "nibabel",
            "nibabel.global",
            "urllib3",
            "PIL",
            "matplotlib",
            "h5py",
            "s3fs",
            "fsspec",
            "botocore",
            "boto3"
        ]
    
    for lib in libraries:
        logging.getLogger(lib).setLevel(logging.WARNING)


@contextmanager
def log_time(logger: logging.Logger, message: str, level: str = "INFO"):
    """
    Context manager for logging execution time.
    
    Args:
        logger: Logger instance
        message: Message describing the operation
        level: Log level (default: INFO)
    
    Yields:
        None
    
    Example:
        >>> with log_time(logger, "Training epoch 1"):
        ...     train_epoch(model, data)
        2024-12-06 10:30:45 - INFO - Training epoch 1...
        2024-12-06 10:35:30 - INFO - Training epoch 1 completed in 4m 45s
    """
    log_func = getattr(logger, level.lower())
    
    log_func(f"{message}...")
    start_time = time.time()
    
    try:
        yield
    finally:
        elapsed = time.time() - start_time
        
        # Format elapsed time
        if elapsed < 60:
            time_str = f"{elapsed:.1f}s"
        elif elapsed < 3600:
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)
            time_str = f"{minutes}m {seconds}s"
        else:
            hours = int(elapsed // 3600)
            minutes = int((elapsed % 3600) // 60)
            time_str = f"{hours}h {minutes}m"
        
        log_func(f"{message} completed in {time_str}")


@contextmanager
def log_memory(logger: logging.Logger, message: str, level: str = "DEBUG"):
    """
    Context manager for logging memory usage.
    
    Requires psutil package.
    
    Args:
        logger: Logger instance
        message: Message describing the operation
        level: Log level (default: DEBUG)
    
    Yields:
        None
    
    Example:
        >>> with log_memory(logger, "Loading dataset"):
        ...     dataset = load_large_dataset()
        2024-12-06 10:30:45 - DEBUG - Loading dataset (memory: 1.2 GB)
        2024-12-06 10:30:50 - DEBUG - Loading dataset completed (memory: 5.8 GB, delta: +4.6 GB)
    """
    try:
        import psutil
        process = psutil.Process()
    except ImportError:
        logger.warning("psutil not available, memory logging disabled")
        yield
        return
    
    log_func = getattr(logger, level.lower())
    
    # Get initial memory
    mem_info_start = process.memory_info()
    mem_mb_start = mem_info_start.rss / 1024 / 1024
    
    log_func(f"{message} (memory: {mem_mb_start:.1f} MB)")
    
    try:
        yield
    finally:
        # Get final memory
        mem_info_end = process.memory_info()
        mem_mb_end = mem_info_end.rss / 1024 / 1024
        delta_mb = mem_mb_end - mem_mb_start
        
        sign = "+" if delta_mb >= 0 else ""
        log_func(
            f"{message} completed "
            f"(memory: {mem_mb_end:.1f} MB, delta: {sign}{delta_mb:.1f} MB)"
        )


class LoggerContext:
    """
    Context manager for temporarily changing logger level.
    
    Useful for debugging specific code sections.
    
    Example:
        >>> logger = setup_logging("my_module", level="INFO")
        >>> logger.info("This will be logged")
        >>> logger.debug("This won't be logged")
        >>> 
        >>> with LoggerContext(logger, level="DEBUG"):
        ...     logger.debug("This will be logged")
        >>> 
        >>> logger.debug("This won't be logged again")
    """
    
    def __init__(self, logger: logging.Logger, level: str):
        """
        Initialize context.
        
        Args:
            logger: Logger to modify
            level: Temporary log level
        """
        self.logger = logger
        self.new_level = getattr(logging, level.upper())
        self.old_level = logger.level
    
    def __enter__(self):
        """Set temporary log level"""
        self.logger.setLevel(self.new_level)
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore original log level"""
        self.logger.setLevel(self.old_level)


def get_logger(name: str) -> logging.Logger:
    """
    Get or create logger with default configuration.
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        Logger instance
    
    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Module initialized")
    """
    return logging.getLogger(name)


def log_dict(logger: logging.Logger, data: Dict[str, Any], title: Optional[str] = None, level: str = "INFO"):
    """
    Log dictionary with pretty formatting.
    
    Args:
        logger: Logger instance
        data: Dictionary to log
        title: Optional title for the output
        level: Log level (default: INFO)
    
    Example:
        >>> log_dict(logger, {"loss": 0.5, "acc": 0.85}, title="Metrics")
        2024-12-06 10:30:45 - INFO - Metrics:
        2024-12-06 10:30:45 - INFO -   loss: 0.5
        2024-12-06 10:30:45 - INFO -   acc: 0.85
    """
    log_func = getattr(logger, level.lower())
    
    if title:
        log_func(f"{title}:")
    
    for key, value in data.items():
        if isinstance(value, float):
            log_func(f"  {key}: {value:.6f}")
        else:
            log_func(f"  {key}: {value}")


def create_experiment_logger(
    experiment_name: str,
    output_dir: Path,
    level: str = "INFO",
    console: bool = True
) -> logging.Logger:
    """
    Create logger for experiment with automatic file naming.
    
    Args:
        experiment_name: Name of experiment
        output_dir: Output directory for logs
        level: Log level
        console: Whether to also log to console
    
    Returns:
        Configured logger
    
    Example:
        >>> logger = create_experiment_logger("mlp_subj01", Path("outputs/mlp"))
        >>> logger.info("Starting experiment")
        # Logs to: outputs/mlp/logs/mlp_subj01_20241206_103045.log
    """
    # Create logs directory
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamped log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{experiment_name}_{timestamp}.log"
    
    # Setup logger
    logger = setup_logging(
        experiment_name,
        level=level,
        log_file=log_file,
        console=console
    )
    
    logger.info(f"Experiment: {experiment_name}")
    logger.info(f"Log file: {log_file}")
    
    return logger
