import logging
from pathlib import Path

# utils.py
def configure_logging(level=logging.INFO):
    logging.basicConfig()
    logging.getLogger().setLevel(level)

def create_directory(path):
    dir = Path(path)
    dir.mkdir(parents=True, exist_ok=True)
    return dir
