"""
Configuration settings for the monetary expressions to JSON converter model.
"""

import os
from pathlib import Path

import torch

# Project paths
ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = ROOT_DIR / "data"
SRC_DIR = ROOT_DIR / "src"
MODELS_DIR = ROOT_DIR / "models"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Dataset paths
TRAIN_DATA_PATH = DATA_DIR / "train.jsonl"
VAL_DATA_PATH = DATA_DIR / "val.jsonl"
TEST_DATA_PATH = DATA_DIR / "test.jsonl"

# Model parameters
MODEL_NAME = "google/flan-t5-small"
MAX_INPUT_LENGTH = 128
MAX_TARGET_LENGTH = 32

# Device configuration (automatic detection)
if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

# Training parameters
BATCH_SIZE = 16
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 10
WARMUP_RATIO = 0.1
GRADIENT_ACCUMULATION_STEPS = 1
EVALUATION_STRATEGY = "steps"
EVAL_STEPS = 500
SAVE_STEPS = 500
LOGGING_STEPS = 100
MAX_GRAD_NORM = 1.0
FP16 = False  # Mixed precision training

# Generation parameters
MAX_NEW_TOKENS = 32
NUM_BEAMS = 1  # Greedy decoding

# Logging
WANDB_PROJECT = "monetary-expressions-to-json"
WANDB_ENTITY = None  # Set to your wandb username or team name
LOG_LEVEL = "INFO"
