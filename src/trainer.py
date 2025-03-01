"""
Training pipeline using the HuggingFace Trainer API.
"""

import json
import logging
import os

import evaluate
import numpy as np
import wandb
from Levenshtein import distance as levenshtein_distance
from transformers import (
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

import config
from src.dataset import prepare_dataset
from src.model import initialize_model, save_model

logger = logging.getLogger(__name__)


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
    early_stopping_patience=3,
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
        early_stopping_patience (int): Number of evaluations with no improvement after which to stop

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
        report_to="wandb" if wandb.run is not None else "none",
        push_to_hub=False,
    )

    return training_args


def is_valid_json(string):
    """
    Check if a string is valid JSON.

    Args:
        string (str): String to check

    Returns:
        bool: True if valid JSON, False otherwise
    """
    try:
        json.loads(string)
        return True
    except (json.JSONDecodeError, ValueError, TypeError):
        return False


def compute_metrics(eval_preds):
    """
    Compute evaluation metrics for the model.

    Args:
        eval_preds (tuple): Tuple of predictions and labels

    Returns:
        dict: Dictionary of metrics including:
            - exact_match: Exact match accuracy
            - json_validity: Percentage of predictions that are valid JSON
            - mean_amount_diff: Mean absolute difference between predicted and actual amounts
            - mean_levenshtein: Mean Levenshtein distance between predictions and labels
    """
    preds, labels = eval_preds

    # Decode predictions
    if isinstance(preds, tuple):
        preds = preds[0]

    # Get tokenizer from model
    decoder_tokenizer = eval_preds.model.tokenizer

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
    levenshtein_distances = []

    for pred, label in zip(decoded_preds, decoded_labels):
        # Calculate Levenshtein distance
        levenshtein_distances.append(levenshtein_distance(pred, label))

        # Check JSON validity
        pred_valid = is_valid_json(pred)
        label_valid = is_valid_json(label)

        if pred_valid:
            valid_json_count += 1

            # If both are valid JSON, calculate amount difference
            if pred_valid and label_valid:
                try:
                    pred_json = json.loads(pred)
                    label_json = json.loads(label)

                    if "amount" in pred_json and "amount" in label_json:
                        pred_amount = float(pred_json["amount"])
                        label_amount = float(label_json["amount"])
                        valid_amounts.append((pred_amount, label_amount))
                except (ValueError, TypeError, KeyError):
                    pass

    # Calculate numeric difference for valid JSON
    if valid_amounts:
        amount_diffs = [abs(pred - label) for pred, label in valid_amounts]
        mean_amount_diff = np.mean(amount_diffs)
    else:
        mean_amount_diff = float("inf")

    # Calculate json validity percentage
    json_validity = valid_json_count / len(decoded_preds) if decoded_preds else 0

    # Calculate mean Levenshtein distance
    mean_levenshtein = np.mean(levenshtein_distances) if levenshtein_distances else float("inf")

    # Prepare and return metrics
    metrics = {
        "exact_match": exact_match_score["exact_match"],
        "json_validity": json_validity,
        "mean_amount_diff": mean_amount_diff,
        "mean_levenshtein": mean_levenshtein,
    }

    # Log metrics to Weights & Biases if enabled
    if wandb.run is not None:
        wandb.log(metrics)

    return metrics


def setup_trainer(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    data_collator,
    output_dir,
    early_stopping_patience=3,
):
    """
    Set up the Seq2SeqTrainer for fine-tuning.

    Args:
        model (transformers.PreTrainedModel): Model to train
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer
        train_dataset (datasets.Dataset): Training dataset
        eval_dataset (datasets.Dataset): Evaluation dataset
        data_collator (transformers.DataCollator): Data collator
        output_dir (str or Path): Directory to save model checkpoints
        early_stopping_patience (int): Number of evaluations with no improvement after which to stop

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
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
    )

    # Attach tokenizer to model for use in compute_metrics
    trainer.model.tokenizer = tokenizer

    return trainer


def train_model(
    model_name=None,
    train_data_path=None,
    val_data_path=None,
    output_dir=None,
    wandb_logging=True,
    early_stopping_patience=3,
):
    """
    Train the model on the provided dataset.

    Args:
        model_name (str): Name of the pretrained model
        train_data_path (str or Path): Path to the training data
        val_data_path (str or Path): Path to the validation data
        output_dir (str or Path): Directory to save model checkpoints
        wandb_logging (bool): Whether to log to Weights & Biases
        early_stopping_patience (int): Number of evaluations with no improvement after which to stop

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
                "warmup_ratio": config.WARMUP_RATIO,
                "weight_decay": config.WEIGHT_DECAY,
                "max_input_length": config.MAX_INPUT_LENGTH,
                "max_target_length": config.MAX_TARGET_LENGTH,
                "early_stopping_patience": early_stopping_patience,
            },
        )

    # Prepare dataset
    logger.info("Preparing datasets...")
    preprocessed_dataset, tokenizer, data_collator = prepare_dataset(
        train_data_path,
        val_data_path,
        model_name,
    )

    # Initialize model
    logger.info(f"Initializing model: {model_name}")
    model = initialize_model(model_name)

    # Set up trainer
    logger.info("Setting up trainer...")
    trainer = setup_trainer(
        model,
        tokenizer,
        preprocessed_dataset["train"],
        preprocessed_dataset["validation"],
        data_collator,
        output_dir,
        early_stopping_patience,
    )

    # Train the model
    logger.info("Starting training...")
    train_result = trainer.train()

    # Log training metrics
    logger.info(f"Training completed. Training metrics: {train_result.metrics}")
    if wandb_logging:
        wandb.log({"train_runtime": train_result.metrics["train_runtime"]})
        wandb.log({"train_samples_per_second": train_result.metrics["train_samples_per_second"]})

    # Evaluate on validation set
    logger.info("Evaluating on validation set...")
    eval_results = trainer.evaluate()
    logger.info(f"Validation metrics: {eval_results}")

    # Save the best model
    best_model_path = os.path.join(output_dir, "best")
    logger.info(f"Saving best model to {best_model_path}")
    save_model(
        trainer.model,
        tokenizer,
        best_model_path,
        metadata={
            "model_name": model_name,
            "train_data_path": str(train_data_path),
            "val_data_path": str(val_data_path),
            "train_metrics": train_result.metrics,
            "eval_metrics": eval_results,
        },
    )

    # Generate training reports
    generate_training_reports(
        trainer,
        tokenizer,
        preprocessed_dataset["validation"],
        os.path.join(output_dir, "reports"),
    )

    # Finish W&B run
    if wandb_logging:
        wandb.finish()

    return best_model_path


def generate_training_reports(trainer, tokenizer, eval_dataset, output_dir, num_examples=10):
    """
    Generate training reports and example predictions.

    Args:
        trainer (transformers.Seq2SeqTrainer): Trained trainer
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer
        eval_dataset (datasets.Dataset): Evaluation dataset
        output_dir (str or Path): Directory to save reports
        num_examples (int): Number of examples to include in the report

    Returns:
        None
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get a subset of examples
    if len(eval_dataset) > num_examples:
        indices = np.random.choice(len(eval_dataset), num_examples, replace=False)
        examples = [eval_dataset[i] for i in indices]
    else:
        examples = [eval_dataset[i] for i in range(len(eval_dataset))]

    # Generate predictions
    inputs = [tokenizer.decode(ex["input_ids"], skip_special_tokens=True) for ex in examples]
    references = [tokenizer.decode(ex["labels"], skip_special_tokens=True) for ex in examples]

    # Get model predictions
    model = trainer.model
    device = model.device

    # Generate predictions
    predictions = []
    for input_text in inputs:
        # Tokenize input
        input_ids = tokenizer(
            input_text, max_length=config.MAX_INPUT_LENGTH, padding="max_length", truncation=True, return_tensors="pt"
        ).input_ids.to(device)

        # Generate prediction
        output_ids = model.generate(
            input_ids, max_length=config.MAX_TARGET_LENGTH, num_beams=config.NUM_BEAMS, early_stopping=True
        )

        # Decode prediction
        prediction = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        predictions.append(prediction)

    # Create report
    report = []
    for i in range(len(inputs)):
        # Evaluate prediction
        is_exact_match = predictions[i] == references[i]
        is_json_valid = is_valid_json(predictions[i]) and is_valid_json(references[i])

        # Calculate Levenshtein distance
        lev_distance = levenshtein_distance(predictions[i], references[i])

        report.append(
            {
                "input": inputs[i],
                "reference": references[i],
                "prediction": predictions[i],
                "exact_match": is_exact_match,
                "valid_json": is_json_valid,
                "levenshtein_distance": lev_distance,
            }
        )

    # Save report to JSON
    report_path = os.path.join(output_dir, "prediction_examples.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Calculate summary statistics
    exact_match_count = sum(1 for ex in report if ex["exact_match"])
    valid_json_count = sum(1 for ex in report if ex["valid_json"])
    avg_levenshtein = np.mean([ex["levenshtein_distance"] for ex in report])

    summary = {
        "num_examples": len(report),
        "exact_match_ratio": exact_match_count / len(report),
        "valid_json_ratio": valid_json_count / len(report),
        "avg_levenshtein_distance": avg_levenshtein,
    }

    # Save summary to JSON
    summary_path = os.path.join(output_dir, "prediction_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Log to W&B if enabled
    if wandb.run is not None:
        wandb.log(summary)

        # Create a W&B Table
        columns = ["input", "reference", "prediction", "exact_match", "valid_json", "levenshtein_distance"]
        data = [[ex[col] for col in columns] for ex in report]
        table = wandb.Table(columns=columns, data=data)
        wandb.log({"prediction_examples": table})

    logger.info(f"Training reports saved to {output_dir}")
    logger.info(f"Summary: {summary}")
