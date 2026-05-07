import logging
import os
from datetime import datetime
from zoneinfo import ZoneInfo

class SingaporeFormatter(logging.Formatter):
    """Custom logging formatter to force timestamps into Singapore Time (SGT)."""
    
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, tz=ZoneInfo("UTC"))
        sgt_dt = dt.astimezone(ZoneInfo("Asia/Singapore"))
        
        if datefmt:
            return sgt_dt.strftime(datefmt)
        return sgt_dt.strftime('%d-%m-%Y %H:%M:%S')


"""Shared logging utilities for the student score prediction pipeline.
Provides a timezone-aware (SGT, UTC+8) logger with simultaneous
console and persistent file output.
"""
def setup_logger(name, log_file="pipeline.log", level=logging.INFO):
    """Initialises and returns a named logger with SGT-aware formatting.

    Args:
        name (str): Logger name, typically the calling module (e.g. 'train').
        log_file (str): Path to the log output file. Defaults to 'pipeline.log'.
        level (int): Logging level. Defaults to logging.INFO.

    Returns:
        logging.Logger: Configured logger instance with console and file handlers.
    """
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    formatter = SingaporeFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
        datefmt='%d-%m-%Y %H:%M:%S SGT'
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger