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

def setup_logger(name, log_file="pipeline.log", level=logging.INFO):
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