#!/usr/bin/env python3
"""
Fine-tune DistilRoBERTa for verifying check amount matches between verbal and decimal representations.
Enhanced with comprehensive metrics logging, calibration analysis, and training dynamics visualization.
"""

import decimal
import json
import logging
import os
import random
from dataclasses import dataclass

import evaluate
import numpy as np
from datasets import Dataset
from sklearn.metrics import brier_score_loss, confusion_matrix
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)

import wandb

# hush warnings related to tokenizer parallelization
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """Arguments pertaining to which model/config/tokenizer we are going to fine-tune."""

    model_name_or_path: str = "distilroberta-base"
    max_length: int = 128


@dataclass
class DataArguments:
    """Arguments pertaining to what data we are going to input our model for training and eval."""

    train_file: str = "data/train.jsonl"
    validation_file: str = "data/val.jsonl"
    test_file: str = "data/test.jsonl"


@dataclass
class CustomTrainingArguments(TrainingArguments):
    """Custom training arguments extending HuggingFace's TrainingArguments with optimal defaults."""

    # Commonly tuned hyperparameters (exposed in shell script)
    output_dir: str = "results"
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 16
    learning_rate: float = 2e-5

    # Optimization parameters
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Training dynamics
    gradient_accumulation_steps: int = 1
    per_device_eval_batch_size: int = 32

    # Logging and evaluation
    logging_strategy: str = "steps"
    logging_steps: int = 10
    eval_strategy: str = "steps"
    eval_steps: int = 100
    save_strategy: str = "steps"
    save_steps: int = 100
    save_total_limit: int = 3

    # Model selection
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "f1"
    greater_is_better: bool = True
    early_stopping_patience: int = 3


def load_jsonl(file_path: str) -> list[dict]:
    """Load JSONL file and return list of dictionaries."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def generate_example(item: dict, label: int = 1) -> tuple[str, str, int]:
    verbal_amount = item["input"]
    decimal_amount = decimal.Decimal(item["amount"])
    decimal_amount = f"{decimal_amount:.2f}"

    return verbal_amount, decimal_amount, label


def generate_negative_example(item: dict, next_item: dict, label: int = 0) -> tuple[str, str, int]:
    """Generate a negative example by mismatching with a randomly selected negative sample.
    Negative examples are generated with controlled perturbations to avoid negative values and extreme differences.
    """
    verbal_amount = item["input"]
    amount = decimal.Decimal(item["amount"])
    next_amount = decimal.Decimal(next_item["amount"])

    rand_value = random.random()
    if rand_value < 0.4:
        wrong_amount = next_amount
    elif rand_value < 0.7:  # 30% chance for small perturbation: 1-10 cents
        delta = decimal.Decimal(random.randint(1, 10)) / 100
        sign = 1 if random.random() < 0.5 else -1
        wrong_amount = amount + sign * delta
    elif rand_value < 0.9:  # 20% chance for moderate perturbation: 1-10 dollars
        delta = decimal.Decimal(random.randint(1, 10))
        sign = 1 if random.random() < 0.5 else -1
        wrong_amount = amount + sign * delta
    else:  # 10% chance for a larger perturbation: 10-100 dollars
        delta = decimal.Decimal(random.randint(10, 100))
        sign = 1 if random.random() < 0.5 else -1
        wrong_amount = amount + sign * delta

    # Ensure the amount is not negative
    if wrong_amount < 0:
        wrong_amount = decimal.Decimal(0)

    wrong_decimal_amount = f"{wrong_amount:.2f}"
    return verbal_amount, wrong_decimal_amount, label


def prepare_dataset(data: list[dict]) -> Dataset:
    """Convert list of dictionaries to HuggingFace Dataset with random pairing for negative examples."""
    processed_data = {"verbal_amount": [], "decimal_amount": [], "label": []}
    n = len(data)
    for i, item in enumerate(data):
        # Generate positive example
        verbal_amount, decimal_amount, label = generate_example(item)
        processed_data["verbal_amount"].append(verbal_amount)
        processed_data["decimal_amount"].append(decimal_amount)
        processed_data["label"].append(label)

        # Generate negative example by selecting a random different item
        j = random.randint(0, n - 1)
        while j == i:
            j = random.randint(0, n - 1)
        negative_item = data[j]
        verbal_amount, decimal_amount, label = generate_negative_example(item, negative_item)
        processed_data["verbal_amount"].append(verbal_amount)
        processed_data["decimal_amount"].append(decimal_amount)
        processed_data["label"].append(label)

    return Dataset.from_dict(processed_data)


def compute_metrics(eval_pred) -> dict[str, float]:
    """Compute metrics for evaluation."""
    predictions, labels = eval_pred
    predictions_labels = np.argmax(predictions, axis=1)

    # Load core metrics from evaluate
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1", zero_division=0)

    accuracy_result = accuracy_metric.compute(predictions=predictions_labels, references=labels)
    f1_result = f1_metric.compute(predictions=predictions_labels, references=labels)

    accuracy = accuracy_result["accuracy"] if accuracy_result is not None else 0.0
    f1 = f1_result["f1"] if f1_result is not None else 0.0

    # Compute confusion matrix
    cm = confusion_matrix(labels, predictions_labels)
    if cm.shape == (2, 2):
        TN, FP, FN, TP = cm.ravel()  # noqa: N806
    else:
        TN, FP, FN, TP = 0, 0, 0, 0  # noqa: N806

    # Compute softmax probabilities for confidence analysis
    exp_preds = np.exp(predictions)
    predictions_softmax = exp_preds / np.sum(exp_preds, axis=1, keepdims=True)

    # Brier score for calibration measurement
    try:
        brier_score = brier_score_loss(labels, predictions_softmax[:, 1])
    except ValueError:
        brier_score = 0.0

    # Create metrics dictionary to log to wandb
    wandb_metrics = {
        # Standard metrics
        "accuracy": float(accuracy),
        "f1": float(f1),
        "brier_score": float(brier_score),
        # Distribution metrics
        "label_distribution": {
            "true_positives": int(TP),
            "true_negatives": int(TN),
            "false_positives": int(FP),
            "false_negatives": int(FN),
        },
        # Confusion Matrix
        "confusion_matrix": wandb.plot.confusion_matrix(probs=None, y_true=labels, preds=predictions_labels, class_names=["Mismatch", "Match"]),
    }

    # Log everything to wandb
    wandb.log(wandb_metrics)

    # Return core metrics for model selection
    return {
        "accuracy": float(accuracy),
        "f1": float(f1),
        "brier_score": float(brier_score),
    }


def main():
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments))  # type: ignore
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Initialize wandb
    wandb.init(
        project="check-amount-verification",
        name=f"distilroberta-check-verifier-{wandb.util.generate_id()}",
        config={**vars(model_args), **vars(data_args), **vars(training_args)},
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    logger.info("Auto-determining max_length based on training data...")
    raw_train_data = load_jsonl(data_args.train_file)
    lengths = []
    for item in raw_train_data:
        # Tokenize without padding to measure true length
        tokenized = tokenizer(
            item["input"],
            str(item["amount"]),
            truncation=True,
            padding=False,
        )
        lengths.append(len(tokenized.get("input_ids", [])))

    # get 95th percentile of lengths
    smart_max_length = int(np.percentile(lengths, 95))
    # round smart_max_length to the nearest multiple of 8 (for batching)
    smart_max_length = round(smart_max_length / 8) * 8
    model_args.max_length = smart_max_length
    logger.info(f"Auto-determined max_length: {smart_max_length}")

    # Load and process datasets
    logger.info("Loading datasets...")
    raw_train_data = load_jsonl(data_args.train_file)
    raw_eval_data = load_jsonl(data_args.validation_file)
    raw_test_data = load_jsonl(data_args.test_file)

    train_data = prepare_dataset(raw_train_data)
    eval_data = prepare_dataset(raw_eval_data)
    test_data = prepare_dataset(raw_test_data)

    # Define preprocess function with dynamic padding support
    def preprocess_function(examples):
        tokenized = tokenizer(
            examples["verbal_amount"],
            examples["decimal_amount"],
            truncation=True,
            max_length=model_args.max_length,
        )

        tokenized["labels"] = examples["label"]
        return tokenized

    # Preprocess datasets
    logger.info("Preprocessing datasets...")
    train_dataset = train_data.map(preprocess_function, batched=True, remove_columns=train_data.column_names)
    eval_dataset = eval_data.map(preprocess_function, batched=True, remove_columns=eval_data.column_names)
    test_dataset = test_data.map(preprocess_function, batched=True, remove_columns=test_data.column_names)

    # Analyze class distribution
    train_labels = train_dataset["labels"]
    eval_labels = eval_dataset["labels"]
    test_labels = test_dataset["labels"]

    train_pos = sum(train_labels)
    train_neg = len(train_labels) - train_pos
    eval_pos = sum(eval_labels)
    eval_neg = len(eval_labels) - eval_pos
    test_pos = sum(test_labels)
    test_neg = len(test_labels) - test_pos

    class_dist_table = wandb.Table(columns=["Split", "Positive (Match)", "Negative (Mismatch)", "Positive %", "Negative %"])
    class_dist_table.add_data("Train", train_pos, train_neg, train_pos / len(train_labels) * 100, train_neg / len(train_labels) * 100)
    class_dist_table.add_data("Validation", eval_pos, eval_neg, eval_pos / len(eval_labels) * 100, eval_neg / len(eval_labels) * 100)
    class_dist_table.add_data("Test", test_pos, test_neg, test_pos / len(test_labels) * 100, test_neg / len(test_labels) * 100)

    # Log dataset statistics
    wandb.log(
        {
            "dataset/train_size": len(train_dataset),
            "dataset/eval_size": len(eval_dataset),
            "dataset/test_size": len(test_dataset),
            "dataset/class_distribution": class_dist_table,
        }
    )

    # Initialize collator
    data_collator = DataCollatorWithPadding(tokenizer)

    # Initialize model
    logger.info("Initializing model...")
    model = AutoModelForSequenceClassification.from_pretrained(model_args.model_name_or_path, num_labels=2)

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=training_args.early_stopping_patience),
        ],
    )

    # Train model
    logger.info("Starting training...")
    trainer.train()

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_results = trainer.evaluate(test_dataset)  # type: ignore

    # Convert test results to a format suitable for logging
    test_metrics = {f"test/{k.replace('eval_', '')}": v for k, v in test_results.items()}
    wandb.log(test_metrics)

    # Log test results as a table
    test_results_table = wandb.Table(columns=["Metric", "Value"])

    for k, v in test_metrics.items():
        test_results_table.add_data(k, v)

    wandb.log({"test/results_summary": test_results_table})

    # Save final model
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

    # Create a model summary
    model_summary = {
        "model_name": model_args.model_name_or_path,
        "task": "Check amount verification",
        "train_samples": len(train_dataset),
        "eval_samples": len(eval_dataset),
        "test_samples": len(test_dataset),
        "epochs": training_args.num_train_epochs,
        "learning_rate": training_args.learning_rate,
        "batch_size": training_args.per_device_train_batch_size,
        "max_length": model_args.max_length,
        "test_accuracy": test_metrics.get("test/accuracy", 0.0),
        "test_f1": test_metrics.get("test/f1", 0.0),
        "test_expected_calibration_error": test_metrics.get("test/expected_calibration_error", 0.0),
    }

    # Log model summary
    wandb.summary.update(model_summary)

    # Close wandb
    wandb.finish()


if __name__ == "__main__":
    main()
