import logging
from logging.handlers import RotatingFileHandler
import os

def get_logger(name: str):
    """Return a logger that logs to console and rotating file."""
    log_dir = "face_recognition/logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "face_recognition.log")

    logger = logging.getLogger(name)
    if logger.handlers:   # avoid duplicate handlers if imported twice
        return logger

    logger.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s:%(name)s: %(message)s"))

    # Rotating file handler (5 MB, keep 3 backups)
    fh = RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=3)
    fh.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s:%(name)s: %(message)s"))

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger
