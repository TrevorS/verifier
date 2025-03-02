"""
Configuration settings for the monetary expressions to JSON converter model.
"""

import logging
import os
from pathlib import Path

import torch

# Configure logger
logger = logging.getLogger(__name__)

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
    logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
elif hasattr(torch, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
    logger.info("Using Apple Silicon MPS (Metal Performance Shaders)")
else:
    DEVICE = "cpu"
    logger.info("No GPU detected, using CPU for training")

# Training parameters
BATCH_SIZE = 16
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 10
WARMUP_RATIO = 0.1
GRADIENT_ACCUMULATION_STEPS = 4
EVAL_STRATEGY = "steps"
EVAL_STEPS = 500
SAVE_STEPS = 500
LOGGING_STEPS = 100
MAX_GRAD_NORM = 1.0
FP16 = False  # Mixed precision training

# Generation parameters
MAX_NEW_TOKENS = 32
NUM_BEAMS = 4  # Using beam search instead of greedy decoding for better results

# Logging
WANDB_PROJECT = "monetary-expressions-to-json"
WANDB_ENTITY = None  # Set to your wandb username or team name
LOG_LEVEL = "INFO"
