import logging
import logging.handlers
from pathlib import Path

def setup_logging (log_level=logging.INFO):
    """Setup application logging"""

    #Create logs directory

    log_dir = Path(__file__).parent.parent.parent / "logs"
    log_dir.mkdir(exist_ok=TRUE)

    #Create logger
    logger = logging.getLogger('heart_disease_api')
    logger.setLevel(log_level)

    # File handler

    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / 'app.log',
        maxBytes = 10485760, #10MB
        backupCount = 10
    )
    file_handler.setLevel(log_level)

    #Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)

    #Formatter
    formatter = logging.Formatter(
        '%(asktime)s - %(name)s - %(levelname)s -%(message)s'
    )

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    #Add handlers

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
