import os
import logging
from datetime import datetime

# Define the log directory and file
LOG_DIR = "C:/Users/DHARSHAN KUMAR B J/Music/heart-disease-prediction/logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f"heart_app_log_{datetime.now().strftime('%Y%m%d')}.log")

# Configure logging
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def log_info(message):
    """Log an info message."""
    print(f"[INFO] {message}")
    logging.info(message)

def log_error(message):
    """Log an error message."""
    print(f"[ERROR] {message}")
    logging.error(message)
