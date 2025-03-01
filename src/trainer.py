"""
Training pipeline using the HuggingFace Trainer API.
"""

import json
import os

import evaluate
import numpy as np
import wandb
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

import config
from src.dataset import prepare_dataset
from src.model import initialize_model, save_model


def configure_training_args(
    output_dir,
    num_train_epochs=None,
    batch_size=None,
    learning_rate=None,
    weight_decay=None,
    warmup_ratio=None,
    logging_steps=None,
    evaluation_strategy=None,
    eval_steps=None,
    save_steps=None,
    gradient_accumulation_steps=None,
    fp16=None,
):
    """
    Configure training arguments for the Seq2SeqTrainer.

    Args:
        output_dir (str or Path): Directory to save model checkpoints
        num_train_epochs (int): Number of training epochs
        batch_size (int): Batch size for training and evaluation
        learning_rate (float): Learning rate
        weight_decay (float): Weight decay
        warmup_ratio (float): Ratio of steps for learning rate warmup
        logging_steps (int): Number of steps between logging updates
        evaluation_strategy (str): Evaluation strategy
        eval_steps (int): Number of steps between evaluations
        save_steps (int): Number of steps between saving checkpoints
        gradient_accumulation_steps (int): Number of steps to accumulate gradients
        fp16 (bool): Whether to use mixed precision training

    Returns:
        transformers.Seq2SeqTrainingArguments: Training arguments
    """
    # Use default values from config if not provided
    if num_train_epochs is None:
        num_train_epochs = config.NUM_EPOCHS

    if batch_size is None:
        batch_size = config.BATCH_SIZE

    if learning_rate is None:
        learning_rate = config.LEARNING_RATE

    if weight_decay is None:
        weight_decay = config.WEIGHT_DECAY

    if warmup_ratio is None:
        warmup_ratio = config.WARMUP_RATIO

    if logging_steps is None:
        logging_steps = config.LOGGING_STEPS

    if evaluation_strategy is None:
        evaluation_strategy = config.EVALUATION_STRATEGY

    if eval_steps is None:
        eval_steps = config.EVAL_STEPS

    if save_steps is None:
        save_steps = config.SAVE_STEPS

    if gradient_accumulation_steps is None:
        gradient_accumulation_steps = config.GRADIENT_ACCUMULATION_STEPS

    if fp16 is None:
        fp16 = config.FP16

    # Configure training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        logging_steps=logging_steps,
        evaluation_strategy=evaluation_strategy,
        eval_steps=eval_steps,
        save_steps=save_steps,
        save_total_limit=2,
        gradient_accumulation_steps=gradient_accumulation_steps,
        predict_with_generate=True,
        generation_max_length=config.MAX_TARGET_LENGTH,
        load_best_model_at_end=True,
        metric_for_best_model="exact_match",
        greater_is_better=True,
        fp16=fp16,
    )

    return training_args


def compute_metrics(eval_preds):
    """
    Compute evaluation metrics for the model.

    Args:
        eval_preds (tuple): Tuple of predictions and labels

    Returns:
        dict: Dictionary of metrics
    """
    preds, labels = eval_preds

    # Decode predictions
    if isinstance(preds, tuple):
        preds = preds[0]

    # Get tokenizer from model
    decoder_tokenizer = Seq2SeqTrainer.model.generation_config.tokenizer

    # Decode predictions and labels
    decoded_preds = decoder_tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = decoder_tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Post-process predictions and labels
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]

    # Initialize metrics
    exact_match = evaluate.load("exact_match")

    # Calculate metrics
    exact_match_score = exact_match.compute(predictions=decoded_preds, references=decoded_labels)

    # Check JSON validity and extract amounts for numeric difference
    valid_json_count = 0
    valid_amounts = []

    for pred, label in zip(decoded_preds, decoded_labels):
        try:
            pred_json = json.loads(pred)
            label_json = json.loads(label)

            if "amount" in pred_json and "amount" in label_json:
                pred_amount = pred_json["amount"]
                label_amount = label_json["amount"]
                valid_amounts.append((pred_amount, label_amount))

            valid_json_count += 1
        except (json.JSONDecodeError, ValueError, TypeError, KeyError):
            pass

    # Calculate numeric difference for valid JSON
    if valid_amounts:
        amount_diffs = [abs(pred - label) for pred, label in valid_amounts]
        mean_amount_diff = np.mean(amount_diffs)
    else:
        mean_amount_diff = float("inf")

    # Calculate json validity percentage
    json_validity = valid_json_count / len(decoded_preds) if decoded_preds else 0

    return {
        "exact_match": exact_match_score["exact_match"],
        "json_validity": json_validity,
        "mean_amount_diff": mean_amount_diff,
    }


def setup_trainer(model, tokenizer, train_dataset, eval_dataset, data_collator, output_dir):
    """
    Set up the Seq2SeqTrainer for fine-tuning.

    Args:
        model (transformers.PreTrainedModel): Model to train
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer
        train_dataset (datasets.Dataset): Training dataset
        eval_dataset (datasets.Dataset): Evaluation dataset
        data_collator (transformers.DataCollator): Data collator
        output_dir (str or Path): Directory to save model checkpoints

    Returns:
        transformers.Seq2SeqTrainer: Configured trainer
    """
    # Configure training arguments
    training_args = configure_training_args(output_dir)

    # Set up trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    return trainer


def train_model(
    model_name=None,
    train_data_path=None,
    val_data_path=None,
    output_dir=None,
    wandb_logging=True,
):
    """
    Train the model on the provided dataset.

    Args:
        model_name (str): Name of the pretrained model
        train_data_path (str or Path): Path to the training data
        val_data_path (str or Path): Path to the validation data
        output_dir (str or Path): Directory to save model checkpoints
        wandb_logging (bool): Whether to log to Weights & Biases

    Returns:
        str: Path to the saved model
    """
    # Use default values from config if not provided
    if model_name is None:
        model_name = config.MODEL_NAME

    if train_data_path is None:
        train_data_path = config.TRAIN_DATA_PATH

    if val_data_path is None:
        val_data_path = config.VAL_DATA_PATH

    if output_dir is None:
        output_dir = config.MODELS_DIR / "checkpoints"

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Initialize W&B
    if wandb_logging:
        wandb.init(
            project=config.WANDB_PROJECT,
            entity=config.WANDB_ENTITY,
            config={
                "model_name": model_name,
                "learning_rate": config.LEARNING_RATE,
                "batch_size": config.BATCH_SIZE,
                "epochs": config.NUM_EPOCHS,
            },
        )

    # Prepare dataset
    preprocessed_dataset, tokenizer, data_collator = prepare_dataset(
        train_data_path,
        val_data_path,
        model_name,
    )

    # Initialize model
    model = initialize_model(model_name)

    # Set up trainer
    trainer = setup_trainer(
        model,
        tokenizer,
        preprocessed_dataset["train"],
        preprocessed_dataset["validation"],
        data_collator,
        output_dir,
    )

    # Train the model
    trainer.train()

    # Save the best model
    best_model_path = os.path.join(output_dir, "best")
    save_model(
        trainer.model,
        tokenizer,
        best_model_path,
        metadata={
            "model_name": model_name,
            "train_data_path": str(train_data_path),
            "val_data_path": str(val_data_path),
        },
    )

    # Finish W&B run
    if wandb_logging:
        wandb.finish()

    return best_model_path
