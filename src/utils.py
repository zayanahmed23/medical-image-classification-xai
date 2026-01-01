# src/utils.py
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Make runs more reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic can be slower, but good for research sanity.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str | Path) -> Path:
    """Create a folder if it doesn't exist and return Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def check_data_layout(data_dir: str | Path) -> dict:
    """
    Validate the ImageFolder layout:
    data/raw/{train,val,test}/{NORMAL,PNEUMONIA}/...
    """
    data_dir = Path(data_dir)
    expected = {
        "train": data_dir / "train",
        "val": data_dir / "val",
        "test": data_dir / "test",
    }
    missing = [k for k, p in expected.items() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing split folder(s): {missing}. Expected under: {data_dir}\n"
            f"Example: {data_dir}/train/NORMAL, {data_dir}/train/PNEUMONIA"
        )
    return {k: str(p) for k, p in expected.items()}


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
