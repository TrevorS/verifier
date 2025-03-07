#!/bin/bash

# Exit on error
set -e

# Configuration
MODEL="distilroberta-base"
NUM_EPOCHS=1
BATCH_SIZE=64
LEARNING_RATE=2e-5
RUN_NAME="$MODEL-$NUM_EPOCHS-epoch-$BATCH_SIZE-batch-$LEARNING_RATE-lr"
OUTPUT_DIR="results/$RUN_NAME"
EVAL_STEPS=500
SAVE_STEPS=500
LOGGING_STEPS=10

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run training
echo "Starting training..."
python scripts/train_distilroberta.py \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs "$NUM_EPOCHS" \
    --per_device_train_batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --model_name_or_path "$MODEL" \
    --run_name "$RUN_NAME" \
    --eval_steps "$EVAL_STEPS" \
    --logging_steps "$LOGGING_STEPS" \
    --save_steps "$SAVE_STEPS" 