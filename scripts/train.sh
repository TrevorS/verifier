#!/bin/bash

# Exit on error
set -e

# Configuration
MODEL="distilroberta-base"
NUM_EPOCHS=5
BATCH_SIZE=128
LEARNING_RATE=2e-5
RUN_NAME="$MODEL-$NUM_EPOCHS-epoch-$BATCH_SIZE-batch-$LEARNING_RATE-lr-max-logging"
OUTPUT_DIR="results/$RUN_NAME"
EVAL_STEPS=1000
SAVE_STEPS=1000
LOGGING_STEPS=20
GRADIENT_ACCUMULATION_STEPS=2

FP16="false"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run training
echo "Starting training..."
python scripts/train.py \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs "$NUM_EPOCHS" \
    --per_device_train_batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --model_name_or_path "$MODEL" \
    --run_name "$RUN_NAME" \
    --eval_steps "$EVAL_STEPS" \
    --logging_steps "$LOGGING_STEPS" \
    --save_steps "$SAVE_STEPS" \
    --fp16 "$FP16" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
