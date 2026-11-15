"""
utils.py
--------
Shared utility functions used across all project modules.

Covers: reproducibility seeding, checkpoint helpers, logging setup,
and progress display.

Phase 1: placeholder stubs — extended in later phases as needed.
"""

import os
import random
import logging
from pathlib import Path


def set_seed(seed: int = 42) -> None:
    """Set random seeds for Python, NumPy, and PyTorch for reproducibility.

    Args:
        seed: Integer seed value. Defaults to config.RANDOM_SEED (42).
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_logger(name: str = __name__, level: int = logging.INFO) -> logging.Logger:
    """Return a module-level logger with a consistent format.

    Args:
        name:  Logger name (use __name__ from the calling module).
        level: Logging level.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def ensure_dir(path) -> Path:
    """Create a directory (and parents) if it does not already exist.

    Args:
        path: str or Path to create.

    Returns:
        Resolved Path object.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_checkpoint(model, tokenizer, save_dir: str) -> None:
    """Save a Hugging Face model and tokenizer to disk.

    Args:
        model:     A Hugging Face PreTrainedModel.
        tokenizer: Corresponding PreTrainedTokenizer.
        save_dir:  Target directory.
    """
    raise NotImplementedError("Implemented when training modules are built.")


def load_checkpoint(model_class, save_dir: str):
    """Load a Hugging Face model from a saved checkpoint directory.

    Args:
        model_class: The model class (e.g. BertForSequenceClassification).
        save_dir:    Path to the checkpoint directory.

    Returns:
        Loaded model instance.
    """
    raise NotImplementedError("Implemented when training modules are built.")
