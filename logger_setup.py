import logging
import sys
from logging.handlers import RotatingFileHandler


def setup_logger():
    """Configure a root logger that works reliably across all files."""
    # Clear existing log file
    open('app.log', 'w').close()  # This erases all previous logs

    # Create a logger
    logger = logging.getLogger('my_app')
    logger.setLevel(logging.DEBUG)

    # File handler will start fresh
    file_handler = logging.FileHandler('app.log', mode='a')  # 'a' for future appends

    logger = logging.getLogger()  #  Get the root logger

    # Completely reset logger if needed (remove all handlers)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    logger.setLevel(logging.DEBUG)  # Set root logger to lowest level

    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # File handler with rotation (5MB, 3 backup files)
    file_handler = RotatingFileHandler(
        'app.log',
        mode='a',
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Keep propagation ON for root logger
    logger.propagate = True

    return logger


#  Initialize immediately when module loads
logger = setup_logger()