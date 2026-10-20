"""Central configuration. Paths can be overridden with environment variables."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Folder that contains train_image_paths.csv, valid_image_paths.csv, train/ and valid/
DATA_ROOT = Path(os.environ.get("MURA_ROOT", PROJECT_ROOT / "data" / "MURA-v1.1"))
OUTPUT_ROOT = Path(os.environ.get("MURA_OUTPUT", PROJECT_ROOT / "outputs"))
# File manifest (columns: file_name, size, md5_hash, ...) listing every MURA file. Optional.
MANIFEST = Path(os.environ.get("MURA_MANIFEST", PROJECT_ROOT / "data" / "mura_v1_1.csv"))

IMG_SIZE = 224          # EfficientNetB0 default input size
SEED = 42
VAL_FRACTION = 0.15     # fraction of *patients* in official train held out for validation


def set_data_root(path):
    global DATA_ROOT
    DATA_ROOT = Path(path)


def set_output_root(path):
    global OUTPUT_ROOT
    OUTPUT_ROOT = Path(path)
