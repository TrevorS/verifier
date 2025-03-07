#!/usr/bin/env python3
"""
Fine-tune DistilRoBERTa for verifying check amount matches between verbal and decimal representations.
"""

import decimal
import json
import logging
import os
import random
from dataclasses import dataclass

import numpy as np
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
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

    # Early stopping
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
    """Generate a negative example by mismatching with next amount."""
    # ---------------------------------------
    # | % chance | type of mismatch         |
    # |----------|--------------------------|
    # | 40%      | next amount              |
    # | 10%      | +/- 1-10 cents           |
    # | 10%      | +/- 1-10 dollars         |
    # | 10%      | +/- 10-100 dollars       |
    # | 10%      | +/- 100-1000 dollars     |
    # | 10%      | +/- 1000-10000 dollars   |
    # | 10%      | +/- 10000-100000 dollars |
    # ---------------------------------------
    verbal_amount = item["input"]
    amount = decimal.Decimal(item["amount"])
    next_amount = decimal.Decimal(next_item["amount"])

    rand_value = random.random()
    if rand_value < 0.4:
        wrong_decimal_amount = next_amount
    elif rand_value < 0.5:
        wrong_decimal_amount = amount + decimal.Decimal(random.randint(-100, 100)) / 100
    elif rand_value < 0.6:
        wrong_decimal_amount = amount + decimal.Decimal(random.randint(-1000, 1000)) / 100
    elif rand_value < 0.7:
        wrong_decimal_amount = amount + decimal.Decimal(random.randint(-10000, 10000)) / 100
    elif rand_value < 0.8:
        wrong_decimal_amount = amount + decimal.Decimal(random.randint(-100000, 100000)) / 100
    elif rand_value < 0.9:
        wrong_decimal_amount = amount + decimal.Decimal(random.randint(-1000000, 1000000)) / 100
    else:
        wrong_decimal_amount = amount + decimal.Decimal(random.randint(-10000000, 10000000)) / 100
    wrong_decimal_amount = f"{wrong_decimal_amount:.2f}"
    return verbal_amount, wrong_decimal_amount, label


def prepare_dataset(data: list[dict]) -> Dataset:
    """Convert list of dictionaries to HuggingFace Dataset."""
    processed_data = {"verbal_amount": [], "decimal_amount": [], "label": []}

    # zip data into pairs, where each pair is the current item and the next item
    for item, next_item in list(zip(data, data[1:] + [data[0]])):
        # generate positive example
        verbal_amount, decimal_amount, label = generate_example(item)
        processed_data["verbal_amount"].append(verbal_amount)
        processed_data["decimal_amount"].append(decimal_amount)
        processed_data["label"].append(label)

        # generate negative example
        verbal_amount, decimal_amount, label = generate_negative_example(item, next_item)
        processed_data["verbal_amount"].append(verbal_amount)
        processed_data["decimal_amount"].append(decimal_amount)
        processed_data["label"].append(label)

    return Dataset.from_dict(processed_data)


def compute_metrics(eval_pred) -> dict[str, float]:
    """Compute metrics for evaluation."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary")
    accuracy = accuracy_score(labels, predictions)

    # Log confusion matrix to wandb
    wandb.log({"conf_mat": wandb.plot.confusion_matrix(probs=None, y_true=labels, preds=predictions, class_names=["Not Match", "Match"])})

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def main():
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Initialize wandb
    wandb.init(
        project="check-amount-verification",
        name="distilroberta-check-verifier",
        config={**vars(model_args), **vars(data_args), **vars(training_args)},
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    # Load and process datasets
    logger.info("Loading datasets...")
    train_data = prepare_dataset(load_jsonl(data_args.train_file))
    eval_data = prepare_dataset(load_jsonl(data_args.validation_file))
    test_data = prepare_dataset(load_jsonl(data_args.test_file))

    def preprocess_function(examples):
        """Tokenize and prepare inputs for the model."""
        tokenized = tokenizer(
            examples["verbal_amount"],
            examples["decimal_amount"],
            padding="max_length",
            truncation=True,
            max_length=model_args.max_length,
        )
        # Include labels in the processed dataset
        tokenized["labels"] = examples["label"]
        return tokenized

    # Preprocess datasets
    logger.info("Preprocessing datasets...")
    train_dataset = train_data.map(preprocess_function, batched=True, remove_columns=train_data.column_names)
    eval_dataset = eval_data.map(preprocess_function, batched=True, remove_columns=eval_data.column_names)
    test_dataset = test_data.map(preprocess_function, batched=True, remove_columns=test_data.column_names)

    # Initialize model
    logger.info("Initializing model...")
    model = AutoModelForSequenceClassification.from_pretrained(model_args.model_name_or_path, num_labels=2)

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # Train model
    logger.info("Starting training...")
    trainer.train()

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_results = trainer.evaluate(test_dataset)
    logger.info(f"Test results: {test_results}")

    # Save final model
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

    # Close wandb
    wandb.finish()


if __name__ == "__main__":
    main()
