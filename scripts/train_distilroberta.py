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

import evaluate
import numpy as np
from datasets import Dataset
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    HfArgumentParser,
    Trainer,
    TrainerCallback,
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
    """Compute comprehensive metrics for evaluation and debugging."""
    predictions, labels = eval_pred
    predictions_labels = np.argmax(predictions, axis=1)

    # Load metrics from evaluate
    accuracy_metric = evaluate.load("accuracy")
    precision_metric = evaluate.load("precision", zero_division=0)
    recall_metric = evaluate.load("recall", zero_division=0)
    f1_metric = evaluate.load("f1", zero_division=0)

    accuracy_result = accuracy_metric.compute(predictions=predictions_labels, references=labels)
    precision_result = precision_metric.compute(predictions=predictions_labels, references=labels)
    recall_result = recall_metric.compute(predictions=predictions_labels, references=labels)
    f1_result = f1_metric.compute(predictions=predictions_labels, references=labels)

    accuracy = accuracy_result["accuracy"] if accuracy_result is not None else 0.0
    precision = precision_result["precision"] if precision_result is not None else 0.0
    recall = recall_result["recall"] if recall_result is not None else 0.0
    f1 = f1_result["f1"] if f1_result is not None else 0.0

    # Compute softmax probabilities
    exp_preds = np.exp(predictions)
    predictions_softmax = exp_preds / np.sum(exp_preds, axis=1, keepdims=True)

    # ROC AUC and PR AUC
    try:
        auc = roc_auc_score(labels, predictions_softmax[:, 1])
        pr_auc = average_precision_score(labels, predictions_softmax[:, 1])
    except ValueError:
        auc = 0.0
        pr_auc = 0.0

    # Detailed confusion matrix metrics
    cm = confusion_matrix(labels, predictions_labels)
    if cm.shape == (2, 2):
        TN, FP, FN, TP = cm.ravel()  # noqa: N806
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        npv = TN / (TN + FN) if (TN + FN) > 0 else 0.0  # Negative Predictive Value
        fnr = FN / (FN + TP) if (FN + TP) > 0 else 0.0  # False Negative Rate
        fpr = FP / (FP + TN) if (FP + TN) > 0 else 0.0  # False Positive Rate
    else:
        specificity = npv = fnr = fpr = 0.0

    # Confidence analysis
    confidence_scores = np.max(predictions_softmax, axis=1)
    mean_confidence = float(np.mean(confidence_scores))
    mean_confidence_correct = float(np.mean(confidence_scores[predictions_labels == labels]))
    mean_confidence_incorrect = float(np.mean(confidence_scores[predictions_labels != labels]))

    # Sample predictions for visualization
    indices = np.arange(len(labels))
    success_indices = indices[predictions_labels == labels]
    failure_indices = indices[predictions_labels != labels]

    # Create wandb tables for success and failure samples
    success_table = wandb.Table(columns=["True Label", "Predicted Label", "Confidence", "Match Probability", "Mismatch Probability"])
    failure_table = wandb.Table(columns=["True Label", "Predicted Label", "Confidence", "Match Probability", "Mismatch Probability"])

    # Random sample of successes and failures
    num_samples = min(5, len(success_indices), len(failure_indices))

    if num_samples > 0:
        success_sample_indices = np.random.choice(success_indices, num_samples, replace=False)
        failure_sample_indices = np.random.choice(failure_indices, num_samples, replace=False)

        for idx in success_sample_indices:
            success_table.add_data(
                "Match" if labels[idx] == 1 else "Mismatch",
                "Match" if predictions_labels[idx] == 1 else "Mismatch",
                float(confidence_scores[idx]),
                float(predictions_softmax[idx, 1]),
                float(predictions_softmax[idx, 0]),
            )

        for idx in failure_sample_indices:
            failure_table.add_data(
                "Match" if labels[idx] == 1 else "Mismatch",
                "Match" if predictions_labels[idx] == 1 else "Mismatch",
                float(confidence_scores[idx]),
                float(predictions_softmax[idx, 1]),
                float(predictions_softmax[idx, 0]),
            )

    # Prediction distribution analysis
    label_distribution = {
        "true_positives": int(TP),
        "true_negatives": int(TN),
        "false_positives": int(FP),
        "false_negatives": int(FN),
    }

    # Log everything to wandb
    wandb.log(
        {
            # Standard metrics
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "auc": float(auc),
            "pr_auc": float(pr_auc),
            # Additional classification metrics
            "specificity": float(specificity),
            "negative_predictive_value": float(npv),
            "false_negative_rate": float(fnr),
            "false_positive_rate": float(fpr),
            # Confidence metrics
            "mean_confidence": mean_confidence,
            "mean_confidence_correct": mean_confidence_correct,
            "mean_confidence_incorrect": mean_confidence_incorrect,
            # Distribution metrics
            "label_distribution": wandb.Table(data=[[k, v] for k, v in label_distribution.items()], columns=["Category", "Count"]),
            # Sample predictions as tables
            "success_samples": success_table,
            "failure_samples": failure_table,
            # Confusion Matrix
            "confusion_matrix": wandb.plot.confusion_matrix(probs=None, y_true=labels, preds=predictions_labels, class_names=["Mismatch", "Match"]),
        }
    )

    # Return core metrics for model selection
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(auc),
        "specificity": float(specificity),
    }


class GradientTrackingCallback(TrainerCallback):
    """Callback to track gradient norms during training."""

    def __init__(self, log_every=100):
        self.log_every = log_every

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.log_every == 0 and model is not None:
            grad_norms = {}
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norms[f"grad_norm/{name}"] = param.grad.norm().item()

            # Compute aggregate statistics
            if grad_norms:
                all_norms = list(grad_norms.values())
                grad_norms["grad_norm/mean"] = np.mean(all_norms)
                grad_norms["grad_norm/median"] = np.median(all_norms)
                grad_norms["grad_norm/max"] = np.max(all_norms)
                grad_norms["grad_norm/min"] = np.min(all_norms)

                wandb.log(grad_norms, step=state.global_step)


def main():
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments))  # type: ignore
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Initialize wandb
    wandb.init(
        project="check-amount-verification",
        name="distilroberta-check-verifier",
        config={**vars(model_args), **vars(data_args), **vars(training_args)},
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    # If auto_max_length flag is enabled, determine smart max length from training data
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
    smart_max_length = int(np.percentile(lengths, 95))
    model_args.max_length = smart_max_length
    logger.info(f"Auto-determined max_length: {smart_max_length}")

    # Load and process datasets
    logger.info("Loading datasets...")
    train_data = prepare_dataset(load_jsonl(data_args.train_file))
    eval_data = prepare_dataset(load_jsonl(data_args.validation_file))
    test_data = prepare_dataset(load_jsonl(data_args.test_file))

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

    # Initialize DataCollatorWithPadding if dynamic padding is enabled
    data_collator = DataCollatorWithPadding(tokenizer)

    # Initialize model
    logger.info("Initializing model...")
    model = AutoModelForSequenceClassification.from_pretrained(model_args.model_name_or_path, num_labels=2)

    # Initialize trainer with additional callbacks
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=training_args.early_stopping_patience),
            GradientTrackingCallback(log_every=training_args.logging_steps),
        ],
    )

    # Train model
    logger.info("Starting training...")
    trainer.train()

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_results = trainer.evaluate(test_dataset)  # type: ignore
    logger.info(f"Test results: {test_results}")

    # Save final model
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

    # Close wandb
    wandb.finish()


if __name__ == "__main__":
    main()
