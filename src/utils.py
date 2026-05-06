import logging
import os
from datetime import datetime
from zoneinfo import ZoneInfo

class SingaporeFormatter(logging.Formatter):
    """Custom logging formatter to force timestamps into Singapore Time (SGT)."""
    
    def formatTime(self, record, datefmt=None):
        """Converts the log record creation time to Asia/Singapore timezone.

        Args:
            record (logging.LogRecord): The log record containing execution details.
            datefmt (str, optional): The string format for the datetime. 
                Defaults to '%d-%m-%Y %H:%M:%S'.

        Returns:
            str: The formatted datetime string in Singapore Time (DD-MM-YYYY).
        """
        # Convert UTC epoch timestamp of the log record to datetime object
        dt = datetime.fromtimestamp(record.created, tz=ZoneInfo("UTC"))
        # Convert UTC to Asia/Singapore (SGT, UTC+8)
        sgt_dt = dt.astimezone(ZoneInfo("Asia/Singapore"))
        
        if datefmt:
            return sgt_dt.strftime(datefmt)
        return sgt_dt.strftime('%d-%m-%Y %H:%M:%S') # Changed to DD-MM-YYYY format

def setup_logger(name, log_file="pipeline.log", level=logging.INFO):
    """Configures a cross-platform logger that outputs to both console and a file.

    Forces all log entry timestamps to Asia/Singapore Standard Time (SGT) 
    using the DD-MM-YYYY standard date format.

    Args:
        name (str): The name of the logger (usually __name__).
        log_file (str): Path to the output log file. Defaults to "pipeline.log".
        level (int): Logging level (e.g., logging.INFO). Defaults to logging.INFO.

    Returns:
        logging.Logger: A configured, timezone-locked logger instance.
    """
    # Ensure directory for log file exists
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Updated format string to reflect %d-%m-%Y
    formatter = SingaporeFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
        datefmt='%d-%m-%Y %H:%M:%S SGT'
    )

    # Handler 1: Console Output
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Handler 2: File Output (Appends to pipeline.log)
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setFormatter(formatter)

    # Build and configure the logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Prevent duplicate handlers if setup is called multiple times across modules
    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger