"""
Pipeline constants and general functions
"""

from pathlib import Path

ROOT_DIR: Path = Path(__file__).parent.parent

DATA_DIR: Path = ROOT_DIR / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw_data"
RAW_DATA_DIR.mkdir(exist_ok=True, parents=True)

PROCESSED_DATA_DIR: Path = DATA_DIR / "processed_data"
PROCESSED_DATA_DIR.mkdir(exist_ok=True, parents=True)

MODELS_DIR = ROOT_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True, parents=True)