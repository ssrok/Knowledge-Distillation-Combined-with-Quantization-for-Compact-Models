"""
config.py
---------
Central configuration for the Knowledge Distillation + Quantization project.
All hyper-parameters, paths, and device settings live here so every module
imports from a single source of truth.
"""

import os
import torch
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root (resolved relative to this file so it works in Colab too)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Directory paths
# ---------------------------------------------------------------------------
DATA_DIR            = PROJECT_ROOT / "data"
RESULTS_DIR         = PROJECT_ROOT / "results"
MODELS_DIR          = PROJECT_ROOT / "models"

TEACHER_CKPT_DIR    = MODELS_DIR / "teacher"
STUDENT_BASE_DIR    = MODELS_DIR / "student_baseline"
STUDENT_DIST_DIR    = MODELS_DIR / "student_distilled"
STUDENT_QUANT_DIR   = MODELS_DIR / "student_quantized"

# ---------------------------------------------------------------------------
# Model identifiers (Hugging Face Hub names)
# ---------------------------------------------------------------------------
TEACHER_MODEL_NAME  = "bert-base-uncased"
STUDENT_MODEL_NAME  = "distilbert-base-uncased"

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
DATASET_NAME        = "glue"
DATASET_CONFIG      = "sst2"
NUM_LABELS          = 2
TEXT_COLUMN         = "sentence"
LABEL_COLUMN        = "label"
ID_COLUMN           = "idx"
MAX_SEQ_LENGTH      = 128
LABEL_NAMES         = {0: "negative", 1: "positive"}

# Processed data paths (portable Parquet / CSV exports)
PROCESSED_DIR       = DATA_DIR / "processed"
TRAIN_PARQUET       = PROCESSED_DIR / "train.parquet"
VAL_PARQUET         = PROCESSED_DIR / "validation.parquet"
TRAIN_CSV           = PROCESSED_DIR / "train.csv"
VAL_CSV             = PROCESSED_DIR / "validation.csv"
METADATA_JSON       = PROCESSED_DIR / "metadata.json"

# Subset mode: set to a positive integer to use only that many examples per
# split (useful for fast debugging).  Set to None for the full dataset.
SUBSET_SIZE         = None   # e.g. 500 for quick runs, None for full

# ---------------------------------------------------------------------------
# Training hyper-parameters
# ---------------------------------------------------------------------------
BATCH_SIZE          = 16
NUM_EPOCHS          = 2          # set to 3 for a longer run
LEARNING_RATE       = 2e-5
WEIGHT_DECAY        = 0.01
WARMUP_RATIO        = 0.1        # fraction of total steps used for warm-up
RANDOM_SEED         = 42

# ---------------------------------------------------------------------------
# Knowledge-distillation hyper-parameters
# ---------------------------------------------------------------------------
TEMPERATURE         = 2.0        # softens the teacher's probability distribution
ALPHA               = 0.5        # weight of distillation loss vs. hard-label loss
                                 # loss = alpha * KL + (1 - alpha) * CE

# ---------------------------------------------------------------------------
# Quantization settings
# ---------------------------------------------------------------------------
QUANTIZATION_DTYPE  = torch.qint8   # dynamic INT8 quantization target dtype

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Logging / misc
# ---------------------------------------------------------------------------
LOGGING_STEPS       = 50
SAVE_STEPS          = 500
EVAL_STEPS          = 500

# ---------------------------------------------------------------------------
# Convenience: pretty-print config (called by setup_verify.py)
# ---------------------------------------------------------------------------
def print_config() -> None:
    """Print key configuration values to stdout."""
    print("=" * 60)
    print("  Project Configuration")
    print("=" * 60)
    print(f"  PROJECT_ROOT   : {PROJECT_ROOT}")
    print(f"  DEVICE         : {DEVICE}")
    print(f"  TEACHER        : {TEACHER_MODEL_NAME}")
    print(f"  STUDENT        : {STUDENT_MODEL_NAME}")
    print(f"  DATASET        : {DATASET_NAME}/{DATASET_CONFIG}")
    print(f"  NUM_LABELS     : {NUM_LABELS}")
    print(f"  MAX_SEQ_LENGTH : {MAX_SEQ_LENGTH}")
    print(f"  PROCESSED_DIR  : {PROCESSED_DIR}")
    print(f"  SUBSET_SIZE    : {SUBSET_SIZE}")
    print(f"  BATCH_SIZE     : {BATCH_SIZE}")
    print(f"  NUM_EPOCHS     : {NUM_EPOCHS}")
    print(f"  LEARNING_RATE  : {LEARNING_RATE}")
    print(f"  TEMPERATURE    : {TEMPERATURE}")
    print(f"  ALPHA          : {ALPHA}")
    print(f"  RANDOM_SEED    : {RANDOM_SEED}")
    print("=" * 60)


if __name__ == "__main__":
    print_config()
