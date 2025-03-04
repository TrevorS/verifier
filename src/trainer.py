"""
Training pipeline using the HuggingFace Trainer API.
"""

import json
import logging
import os

import evaluate
import numpy as np
import torch
from Levenshtein import distance as levenshtein_distance
from transformers import (
    AutoTokenizer,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

import config
import wandb
from src.dataset import prepare_dataset
from src.inference import extract_amount
from src.model import initialize_model, save_model

# Set environment variable to avoid tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configure logger
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
    bf16=None,
    early_stopping_patience=None,
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
        fp16 (bool): Whether to use mixed precision training with FP16
        bf16 (bool): Whether to use mixed precision training with BF16
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
        # Use the new parameter name if available, otherwise fallback
        evaluation_strategy = getattr(config, "EVAL_STRATEGY", getattr(config, "EVALUATION_STRATEGY", "steps"))

    if eval_steps is None:
        eval_steps = config.EVAL_STEPS

    if save_steps is None:
        save_steps = config.SAVE_STEPS

    if gradient_accumulation_steps is None:
        gradient_accumulation_steps = config.GRADIENT_ACCUMULATION_STEPS

    if fp16 is None:
        fp16 = config.FP16

    # Add BF16 support
    if bf16 is None:
        bf16 = config.BF16

    # early_stopping_patience is just passed through, not used in Seq2SeqTrainingArguments
    # It will be used later when creating the EarlyStoppingCallback

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
        eval_strategy=evaluation_strategy,
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
        bf16=bf16,
        report_to="wandb" if wandb.run is not None else "none",
        push_to_hub=False,
        run_name=f"finetuned-{os.path.basename(output_dir)}-{wandb.util.generate_id()}" if wandb.run is not None else None,
    )

    logger.info(f"Training configuration: {num_train_epochs} epochs, batch size {batch_size}, eval steps {eval_steps}")
    logger.info(f"Precision settings: FP16={fp16}, BF16={bf16}")

    return training_args


def compute_metrics(eval_preds):
    """
    Compute evaluation metrics for the model.

    Args:
        eval_preds (tuple): Tuple of predictions and labels

    Returns:
        dict: Dictionary of metrics including:
            - exact_match: Exact match accuracy
            - mean_amount_diff: Mean absolute difference between predicted and actual amounts
            - mean_levenshtein: Mean Levenshtein distance between predictions and labels
    """
    preds, labels = eval_preds

    # Decode predictions
    if isinstance(preds, tuple):
        preds = preds[0]

    # Access the tokenizer globally (we'll make sure it's accessible)
    try:
        # First try to get it from the trainer's model if available
        decoder_tokenizer = compute_metrics.tokenizer
    except AttributeError:
        # Fall back to initializing a new tokenizer if not available
        logger.info("Tokenizer not found on compute_metrics function, initializing a new one")
        decoder_tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
        # Store tokenizer for future use
        compute_metrics.tokenizer = decoder_tokenizer

    # Ensure tensors are on CPU for decoding
    if hasattr(preds, "device") and str(preds.device) != "cpu":
        preds = preds.cpu()

    if hasattr(labels, "device") and str(labels.device) != "cpu":
        labels = labels.cpu()

    # Convert to list format first
    preds_list = preds.tolist()
    labels_list = labels.tolist()

    # Safely decode by filtering out invalid token IDs
    try:
        # Filter out any out-of-range token IDs
        max_token_id = decoder_tokenizer.vocab_size

        # Clean prediction token IDs
        clean_preds = []
        for seq in preds_list:
            clean_seq = [token_id for token_id in seq if 0 <= token_id < max_token_id]
            clean_preds.append(clean_seq)

        # Clean label token IDs
        clean_labels = []
        for seq in labels_list:
            clean_seq = [token_id for token_id in seq if 0 <= token_id < max_token_id]
            clean_labels.append(clean_seq)

        # Decode predictions and labels
        decoded_preds = decoder_tokenizer.batch_decode(clean_preds, skip_special_tokens=True)
        decoded_labels = decoder_tokenizer.batch_decode(clean_labels, skip_special_tokens=True)
    except Exception as e:
        logger.error(f"Error during tokenizer decoding: {e}")
        logger.error("Falling back to empty strings for predictions and labels")
        decoded_preds = ["" for _ in range(len(preds_list))]
        decoded_labels = ["" for _ in range(len(labels_list))]

    # Post-process predictions and labels
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]

    # Log a few examples for debugging
    example_data = []
    if len(decoded_preds) > 0:
        num_samples = min(10, len(decoded_preds))
        logger.info(f"Sample predictions (first {num_samples}):")
        for i in range(num_samples):
            pred_amount = extract_amount(decoded_preds[i])
            label_amount = extract_amount(decoded_labels[i])
            logger.info(f"  Pred: '{decoded_preds[i]}' → {pred_amount}")
            logger.info(f"  Label: '{decoded_labels[i]}' → {label_amount}")

            # Collect example data for wandb
            example_data.append(
                {
                    "prediction": decoded_preds[i],
                    "reference": decoded_labels[i],
                    "pred_amount": str(pred_amount) if pred_amount is not None else "None",
                    "label_amount": str(label_amount) if label_amount is not None else "None",
                    "is_exact_match": decoded_preds[i] == decoded_labels[i],
                    "amount_diff": abs(pred_amount - label_amount) if pred_amount is not None and label_amount is not None else None,
                    "levenshtein": levenshtein_distance(decoded_preds[i], decoded_labels[i]),
                }
            )

    # Initialize metrics
    exact_match = evaluate.load("exact_match")

    # Calculate metrics
    exact_match_score = exact_match.compute(predictions=decoded_preds, references=decoded_labels)

    valid_amounts = []
    levenshtein_distances = []
    amount_diffs = []
    exact_matches = []

    for pred, label in zip(decoded_preds, decoded_labels):
        # Calculate Levenshtein distance
        lev_dist = levenshtein_distance(pred, label)
        levenshtein_distances.append(lev_dist)

        # Track exact matches
        exact_matches.append(1 if pred == label else 0)

        # extract amounts
        pred_amount = extract_amount(pred)
        label_amount = extract_amount(label)

        # if both amounts are valid, calculate amount difference
        if pred_amount is not None and label_amount is not None:
            diff = abs(pred_amount - label_amount)
            valid_amounts.append((pred_amount, label_amount))
            amount_diffs.append(diff)

    # Calculate numeric difference for valid amounts
    if valid_amounts:
        mean_amount_diff = np.mean(amount_diffs)
        median_amount_diff = np.median(amount_diffs)
        max_amount_diff = np.max(amount_diffs)
        logger.info(f"Found {len(valid_amounts)} valid amount pairs out of {len(decoded_preds)} examples")
        logger.info(f"Amount diff stats - Mean: {mean_amount_diff:.4f}, Median: {median_amount_diff:.4f}, Max: {max_amount_diff:.4f}")
    else:
        mean_amount_diff = float("inf")
        median_amount_diff = float("inf")
        max_amount_diff = float("inf")
        logger.warning("No valid amount pairs found! Check extract_amount function and model outputs.")

    # Calculate mean Levenshtein distance
    if levenshtein_distances:
        mean_levenshtein = np.mean(levenshtein_distances)
        median_levenshtein = np.median(levenshtein_distances)
        max_levenshtein = np.max(levenshtein_distances)
    else:
        mean_levenshtein = float("inf")
        median_levenshtein = float("inf")
        max_levenshtein = float("inf")

    # Calculate percentage of valid amounts extracted
    valid_extraction_rate = len(valid_amounts) / len(decoded_preds) if decoded_preds else 0

    # Prepare and return metrics
    metrics = {
        "exact_match": exact_match_score["exact_match"],
        "mean_amount_diff": mean_amount_diff,
        "median_amount_diff": median_amount_diff,
        "max_amount_diff": max_amount_diff,
        "mean_levenshtein": mean_levenshtein,
        "median_levenshtein": median_levenshtein,
        "max_levenshtein": max_levenshtein,
        "valid_amount_pairs": len(valid_amounts),
        "total_examples": len(decoded_preds),
        "valid_extraction_rate": valid_extraction_rate,
    }

    # Log metrics to Weights & Biases if enabled
    if wandb.run is not None:
        # Log all metrics
        wandb.log(metrics)

        # Create a histogram of amount differences if we have valid amounts
        if amount_diffs:
            wandb.log({"amount_diff_histogram": wandb.Histogram(amount_diffs)})

        # Create a histogram of Levenshtein distances
        if levenshtein_distances:
            wandb.log({"levenshtein_histogram": wandb.Histogram(levenshtein_distances)})

        # Log example predictions as a table
        if example_data:
            # Create a wandb Table directly from the list of dictionaries
            examples_table = wandb.Table(
                columns=["prediction", "reference", "pred_amount", "label_amount", "is_exact_match", "amount_diff", "levenshtein"],
                data=[
                    [
                        ex["prediction"],
                        ex["reference"],
                        ex["pred_amount"],
                        ex["label_amount"],
                        ex["is_exact_match"],
                        ex["amount_diff"],
                        ex["levenshtein"],
                    ]
                    for ex in example_data
                ],
            )
            wandb.log({"example_predictions": examples_table})

        # Log confusion matrix of exact matches (binary classification)
        if exact_matches:
            wandb.log({"exact_match_dist": wandb.Histogram(exact_matches)})

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
    training_args = configure_training_args(output_dir, early_stopping_patience=early_stopping_patience)

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
    quick_test=False,
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
        quick_test (bool): If True, runs a quick test with a small subset of data and reduced steps
        bf16 (bool): Whether to use BF16 mixed precision training

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

    # Set quick test parameters if enabled
    training_args_kwargs = {}
    if quick_test:
        logger.info("🧪 Running in quick test mode with reduced dataset and parameters")
        training_args_kwargs = {
            "num_train_epochs": 1,
            "batch_size": 4,
            "logging_steps": 5,
            "evaluation_strategy": "steps",
            "eval_steps": 10,
            "save_steps": 10,
        }

    # Initialize W&B
    if wandb_logging:
        # Create a more descriptive run name
        run_name = f"{os.path.basename(model_name)}-{wandb.util.generate_id()}"
        if quick_test:
            run_name = f"quick-test-{run_name}"

        # Collect all hyperparameters
        wandb_config = {
            # Model info
            "model_name": model_name,
            "model_type": model_name.split("/")[-1] if "/" in model_name else model_name,
            # Training hyperparameters
            "learning_rate": config.LEARNING_RATE,
            "batch_size": training_args_kwargs.get("batch_size", config.BATCH_SIZE),
            "epochs": training_args_kwargs.get("num_train_epochs", config.NUM_EPOCHS),
            "warmup_ratio": config.WARMUP_RATIO,
            "weight_decay": config.WEIGHT_DECAY,
            "gradient_accumulation_steps": config.GRADIENT_ACCUMULATION_STEPS,
            "fp16": config.FP16,
            "early_stopping_patience": early_stopping_patience,
            # Generation parameters
            "max_input_length": config.MAX_INPUT_LENGTH,
            "max_target_length": config.MAX_TARGET_LENGTH,
            "num_beams": config.NUM_BEAMS,
            # Data info
            "train_data_path": str(train_data_path),
            "val_data_path": str(val_data_path),
            # Environment info
            "device": config.DEVICE,
            "quick_test": quick_test,
        }

        # Initialize wandb
        wandb.init(
            project=config.WANDB_PROJECT,
            entity=config.WANDB_ENTITY,
            name=run_name,
            config=wandb_config,
        )

    # Prepare dataset
    logger.info("Preparing datasets...")
    preprocessed_dataset, tokenizer, data_collator = prepare_dataset(
        train_data_path,
        val_data_path,
        model_name,
    )

    # Log dataset statistics to wandb
    if wandb_logging:
        wandb.log(
            {
                "train_dataset_size": len(preprocessed_dataset["train"]),
                "val_dataset_size": len(preprocessed_dataset["validation"]),
            }
        )

        # Log a few examples from the training data
        if len(preprocessed_dataset["train"]) > 0:
            train_examples = []
            num_examples = min(10, len(preprocessed_dataset["train"]))
            for i in range(num_examples):
                example = preprocessed_dataset["train"][i]
                # Decode the input_ids and labels
                input_text = tokenizer.decode(example["input_ids"], skip_special_tokens=True)
                target_text = tokenizer.decode(example["labels"], skip_special_tokens=True)
                train_examples.append(
                    {
                        "input": input_text,
                        "target": target_text,
                    }
                )

            # Log examples as a table
            examples_table = wandb.Table(columns=["input", "target"], data=[[ex["input"], ex["target"]] for ex in train_examples])
            wandb.log({"train_examples": examples_table})

    # Limit dataset size for quick test
    if quick_test:
        # Take a small subset of the data for a quick test
        train_size = min(100, len(preprocessed_dataset["train"]))
        val_size = min(20, len(preprocessed_dataset["validation"]))

        # Create smaller datasets
        preprocessed_dataset["train"] = preprocessed_dataset["train"].select(range(train_size))
        preprocessed_dataset["validation"] = preprocessed_dataset["validation"].select(range(val_size))

        logger.info(f"Using {train_size} training examples and {val_size} validation examples for quick test")

    # Initialize model
    logger.info(f"Initializing model: {model_name}")
    model = initialize_model(model_name)

    # Log model architecture to wandb
    if wandb_logging:
        # Count model parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        wandb.log(
            {
                "model_total_parameters": total_params,
                "model_trainable_parameters": trainable_params,
                "model_frozen_parameters": total_params - trainable_params,
            }
        )

    # Store tokenizer for compute_metrics function
    compute_metrics.tokenizer = tokenizer

    # Set up trainer
    logger.info("Setting up trainer...")
    training_args = configure_training_args(output_dir, **training_args_kwargs)

    # Create trainer with updated parameters to avoid deprecation warnings
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": preprocessed_dataset["train"],
        "eval_dataset": preprocessed_dataset["validation"],
        "data_collator": data_collator,
        "compute_metrics": compute_metrics,
        "callbacks": [EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
    }

    # Store the tokenizer so we can use it for compute_metrics
    compute_metrics.tokenizer = tokenizer
    trainer_kwargs["tokenizer"] = tokenizer

    # Create the trainer
    trainer = Seq2SeqTrainer(**trainer_kwargs)

    # Train the model
    logger.info("Starting training...")
    train_result = trainer.train()

    # Log training metrics
    logger.info(f"Training completed. Training metrics: {train_result.metrics}")
    if wandb_logging:
        # Log final training metrics
        wandb.log(
            {
                "train_runtime": train_result.metrics["train_runtime"],
                "train_samples_per_second": train_result.metrics["train_samples_per_second"],
                "train_steps_per_second": train_result.metrics["train_steps_per_second"],
                "train_loss": train_result.metrics.get("train_loss", float("nan")),
                "training_complete": True,
            }
        )

    # Evaluate on validation set
    logger.info("Evaluating on validation set...")
    eval_results = trainer.evaluate()
    logger.info(f"Validation metrics: {eval_results}")

    # Log final evaluation metrics to wandb
    if wandb_logging:
        wandb.log({"final_" + k: v for k, v in eval_results.items()})

    # Save the best model
    best_model_path = os.path.join(output_dir, "best")
    logger.info(f"Saving best model to {best_model_path}")

    # Prepare metadata
    model_metadata = {
        "model_name": model_name,
        "train_data_path": str(train_data_path),
        "val_data_path": str(val_data_path),
        "train_metrics": train_result.metrics,
        "eval_metrics": eval_results,
        "training_args": training_args.to_dict(),
        "timestamp": wandb.util.generate_id(),  # Use as a unique identifier
    }

    save_model(
        trainer.model,
        tokenizer,
        best_model_path,
        metadata=model_metadata,
    )

    # Log model metadata to wandb
    if wandb_logging:
        wandb.log({"model_metadata": model_metadata})

    # Generate training reports
    try:
        logger.info("Generating training reports...")
        reports_dir = os.path.join(output_dir, "reports")
        generate_training_reports(
            trainer,
            tokenizer,
            preprocessed_dataset["validation"],
            reports_dir,
        )

        # Log reports to wandb if available
        if wandb_logging:
            try:
                report_path = os.path.join(reports_dir, "prediction_examples.json")
                summary_path = os.path.join(reports_dir, "prediction_summary.json")

                if os.path.exists(report_path):
                    with open(report_path, "r") as f:
                        report_data = json.load(f)

                    # Create a wandb Table directly from the JSON data
                    if report_data and len(report_data) > 0:
                        # Get column names from the first item
                        columns = list(report_data[0].keys())
                        # Extract data rows
                        data = [[item.get(col, "") for col in columns] for item in report_data]
                        examples_table = wandb.Table(columns=columns, data=data)
                        wandb.log({"prediction_examples_json": examples_table})

                if os.path.exists(summary_path):
                    with open(summary_path, "r") as f:
                        summary_data = json.load(f)
                    wandb.log({"prediction_summary": summary_data})
            except Exception as e:
                logger.error(f"Failed to log reports to wandb: {e}")

    except Exception as e:
        logger.error(f"Failed to generate training reports: {e}")
        logger.error("Continuing without reports generation")

    # Finish W&B run
    if wandb_logging:
        # Log final status
        wandb.log({"training_status": "completed"})
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
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Get a subset of examples
        if len(eval_dataset) > num_examples:
            # Convert numpy indices to native Python integers
            indices = np.random.choice(len(eval_dataset), num_examples, replace=False)
            indices = [int(i) for i in indices]  # Convert numpy.int64 to Python int
            examples = [eval_dataset[i] for i in indices]
        else:
            examples = [eval_dataset[i] for i in range(len(eval_dataset))]

        # Generate predictions
        inputs = [tokenizer.decode(ex["input_ids"], skip_special_tokens=True) for ex in examples]
        references = [tokenizer.decode(ex["labels"], skip_special_tokens=True) for ex in examples]

        # Get model predictions
        model = trainer.model
        device = model.device
        logger.info(f"Generating example predictions on {device}")

        # Generate predictions
        predictions = []
        for input_text in inputs:
            try:
                # Tokenize input
                input_ids = tokenizer(
                    input_text, max_length=config.MAX_INPUT_LENGTH, padding="max_length", truncation=True, return_tensors="pt"
                ).input_ids.to(device)

                # Generate prediction
                with torch.no_grad():
                    # Use config's NUM_BEAMS value and ensure early_stopping is only set when using beam search
                    generation_kwargs = {
                        "max_length": config.MAX_TARGET_LENGTH,
                        "num_beams": config.NUM_BEAMS,
                    }

                    # Only add early stopping for beam search
                    if config.NUM_BEAMS > 1:
                        generation_kwargs["early_stopping"] = True

                    output_ids = model.generate(input_ids, **generation_kwargs)

                # Move output back to CPU for decoding
                output_ids = output_ids.cpu()

                # Decode prediction
                prediction = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                predictions.append(prediction)
            except Exception as e:
                logger.error(f"Error generating prediction for input: {input_text}")
                logger.error(f"Error details: {str(e)}")
                predictions.append("")

        # Create report
        report = []
        for i in range(len(inputs)):
            try:
                # Evaluate prediction
                is_exact_match = predictions[i] == references[i]

                # Calculate Levenshtein distance
                lev_distance = levenshtein_distance(predictions[i], references[i])

                report.append(
                    {
                        "input": inputs[i],
                        "reference": references[i],
                        "prediction": predictions[i],
                        "exact_match": is_exact_match,
                        "levenshtein_distance": lev_distance,
                    }
                )
            except Exception as e:
                logger.error(f"Error creating report entry for index {i}")
                logger.error(f"Error details: {str(e)}")
                # Add a simplified report entry
                report.append(
                    {
                        "input": inputs[i],
                        "reference": references[i],
                        "prediction": predictions[i] if i < len(predictions) else "",
                        "error": str(e),
                    }
                )

        # Save report to JSON
        report_path = os.path.join(output_dir, "prediction_examples.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        # Calculate summary statistics
        try:
            exact_match_count = sum(1 for ex in report if ex.get("exact_match", False))

            # Only include examples without errors for levenshtein distance
            levenshtein_distances = [ex["levenshtein_distance"] for ex in report if "levenshtein_distance" in ex]
            avg_levenshtein = np.mean(levenshtein_distances) if levenshtein_distances else float("inf")

            summary = {
                "num_examples": len(report),
                "exact_match_ratio": exact_match_count / len(report) if report else 0,
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
                columns = ["input", "reference", "prediction", "exact_match", "levenshtein_distance"]
                data = []
                for ex in report:
                    if all(col in ex for col in columns):
                        data.append([ex[col] for col in columns])

                if data:
                    table = wandb.Table(columns=columns, data=data)
                    wandb.log({"prediction_examples": table})

            logger.info(f"Training reports saved to {output_dir}")
            logger.info(f"Summary: {summary}")
        except Exception as e:
            logger.error(f"Error calculating summary statistics: {str(e)}")
            logger.info(f"Training reports (without summary) saved to {output_dir}")
    except Exception as e:
        logger.error(f"Error generating training reports: {str(e)}")
        logger.error("Training will continue, but reports could not be generated")
