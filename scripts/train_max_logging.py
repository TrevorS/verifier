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
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve
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
    # Add control over tracked samples
    track_sample_count: int = 20  # Number of samples to track throughout training
    error_sample_count: int = 10  # Number of error samples to log


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

    # Metrics logging enhancements
    calibration_bins: int = 10  # Number of bins for calibration curve
    log_model_architecture: bool = True  # Whether to log model architecture
    track_learning_dynamics: bool = True  # Whether to track learning dynamics
    detailed_error_analysis: bool = True  # Whether to perform detailed error analysis
    log_attention: bool = False  # Whether to log attention patterns (computationally expensive)


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

    # Handle cases where all predictions are correct or incorrect
    mean_confidence_correct = 0.0
    if np.any(predictions_labels == labels):
        mean_confidence_correct = float(np.mean(confidence_scores[predictions_labels == labels]))

    mean_confidence_incorrect = 0.0
    if np.any(predictions_labels != labels):
        mean_confidence_incorrect = float(np.mean(confidence_scores[predictions_labels != labels]))

    # ========== Enhanced metrics section start ==========

    # Brier score for calibration measurement
    try:
        brier_score = brier_score_loss(labels, predictions_softmax[:, 1])
    except ValueError:
        brier_score = 0.0

    # Calculate calibration curve
    try:
        prob_true, prob_pred = calibration_curve(labels, predictions_softmax[:, 1], n_bins=10)
        calibration_data = [[float(x), float(y)] for x, y in zip(prob_pred, prob_true)]
    except ValueError:
        calibration_data = []

    # Generate ROC curve data for visualization
    try:
        fpr_curve, tpr_curve, _ = roc_curve(labels, predictions_softmax[:, 1])
        roc_data = [[float(x), float(y)] for x, y in zip(fpr_curve, tpr_curve)]
    except ValueError:
        roc_data = []

    # Generate PR curve data for visualization
    try:
        precision_curve, recall_curve, _ = precision_recall_curve(labels, predictions_softmax[:, 1])
        pr_curve_data = [[float(x), float(y)] for x, y in zip(recall_curve, precision_curve)]
    except ValueError:
        pr_curve_data = []

    # Error analysis
    error_indices = np.where(predictions_labels != labels)[0]
    correct_indices = np.where(predictions_labels == labels)[0]

    # Confidence distribution in bins
    confidence_bins = np.linspace(0, 1, 11)  # 10 bins
    confidence_histogram = np.histogram(confidence_scores, bins=confidence_bins)[0].tolist()

    # Bin confidence values by correctness
    if len(correct_indices) > 0:
        correct_conf_hist = np.histogram(confidence_scores[correct_indices], bins=confidence_bins)[0].tolist()
    else:
        correct_conf_hist = [0] * 10

    if len(error_indices) > 0:
        error_conf_hist = np.histogram(confidence_scores[error_indices], bins=confidence_bins)[0].tolist()
    else:
        error_conf_hist = [0] * 10

    # Identify high confidence errors (most concerning)
    high_conf_errors = []
    if len(error_indices) > 0:
        error_confs = confidence_scores[error_indices]
        sorted_error_indices = error_indices[np.argsort(-error_confs)]

        # Take top 10 highest confidence errors
        for idx in sorted_error_indices[: min(10, len(sorted_error_indices))]:
            high_conf_errors.append(
                {
                    "true_label": int(labels[idx]),
                    "pred_label": int(predictions_labels[idx]),
                    "confidence": float(confidence_scores[idx]),
                    "match_prob": float(predictions_softmax[idx, 1]),
                    "mismatch_prob": float(predictions_softmax[idx, 0]),
                    "index": int(idx),
                }
            )

    # Error pattern analysis - find common types of errors
    error_patterns = []
    if len(error_indices) > 0:
        # Count true label -> predicted label transitions
        error_transitions = {}
        for idx in error_indices:
            key = f"{labels[idx]}->{predictions_labels[idx]}"
            error_transitions[key] = error_transitions.get(key, 0) + 1

        # Convert to list of patterns
        for trans, count in error_transitions.items():
            true_label, pred_label = trans.split("->")
            error_patterns.append(
                {
                    "true_label": int(true_label),
                    "pred_label": int(pred_label),
                    "count": count,
                    "percentage": float(count / len(error_indices) * 100),
                }
            )

        # Sort by count
        error_patterns.sort(key=lambda x: x["count"], reverse=True)

    # ========== Enhanced metrics section end ==========

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

    # Create metrics dictionary to log to wandb
    wandb_metrics = {
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
        # New calibration metrics
        "brier_score": float(brier_score),
        # Distribution metrics
        "label_distribution": wandb.Table(data=[[k, v] for k, v in label_distribution.items()], columns=["Category", "Count"]),
        # Sample predictions as tables
        "success_samples": success_table,
        "failure_samples": failure_table,
        # Confusion Matrix
        "confusion_matrix": wandb.plot.confusion_matrix(probs=None, y_true=labels, preds=predictions_labels, class_names=["Mismatch", "Match"]),
    }

    # Add enhanced visualization data
    if calibration_data:
        calibration_table = wandb.Table(data=calibration_data, columns=["Predicted Probability", "True Probability"])
        wandb_metrics["calibration_curve"] = wandb.plot.line(
            calibration_table, "Predicted Probability", "True Probability", title="Calibration Curve"
        )

    if roc_data:
        roc_table = wandb.Table(data=roc_data, columns=["False Positive Rate", "True Positive Rate"])
        wandb_metrics["roc_curve"] = wandb.plot.line(roc_table, "False Positive Rate", "True Positive Rate", title="ROC Curve")

    if pr_curve_data:
        pr_curve_table = wandb.Table(data=pr_curve_data, columns=["Recall", "Precision"])
        wandb_metrics["pr_curve"] = wandb.plot.line(pr_curve_table, "Recall", "Precision", title="Precision-Recall Curve")

    # Confidence histograms
    wandb_metrics["confidence_histogram"] = wandb.Histogram(confidence_scores)

    # Add confidence distribution by correctness
    conf_dist_table = wandb.Table(columns=["Bin Center", "All", "Correct", "Incorrect"])
    bin_centers = (confidence_bins[:-1] + confidence_bins[1:]) / 2

    for i, center in enumerate(bin_centers):
        conf_dist_table.add_data(float(center), confidence_histogram[i], correct_conf_hist[i], error_conf_hist[i])

    wandb_metrics["confidence_distribution"] = wandb.plot.bar(
        conf_dist_table, "Bin Center", ["All", "Correct", "Incorrect"], title="Confidence Distribution"
    )

    # Add high confidence errors table if any
    if high_conf_errors:
        high_conf_error_table = wandb.Table(columns=["Index", "True Label", "Predicted Label", "Confidence", "Match Prob", "Mismatch Prob"])

        for error in high_conf_errors:
            high_conf_error_table.add_data(
                error["index"],
                "Match" if error["true_label"] == 1 else "Mismatch",
                "Match" if error["pred_label"] == 1 else "Mismatch",
                error["confidence"],
                error["match_prob"],
                error["mismatch_prob"],
            )

        wandb_metrics["high_confidence_errors"] = high_conf_error_table

    # Add error patterns table if any
    if error_patterns:
        error_pattern_table = wandb.Table(columns=["True Label", "Predicted Label", "Count", "Percentage"])

        for pattern in error_patterns:
            error_pattern_table.add_data(
                "Match" if pattern["true_label"] == 1 else "Mismatch",
                "Match" if pattern["pred_label"] == 1 else "Mismatch",
                pattern["count"],
                pattern["percentage"],
            )

        wandb_metrics["error_patterns"] = error_pattern_table

    # Log everything to wandb
    wandb.log(wandb_metrics)

    # Return core metrics for model selection
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(auc),
        "specificity": float(specificity),
        "brier_score": float(brier_score),  # Add calibration metric
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


class ExampleTrackingCallback(TrainerCallback):
    """Callback to track specific examples throughout training."""

    def __init__(self, eval_dataset, tokenizer, tracked_indices, log_every=1):
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.tracked_indices = tracked_indices
        self.log_every = log_every
        self.history = {idx: [] for idx in tracked_indices}
        self.raw_examples = {}

        # Store the raw examples for later reference
        for idx in tracked_indices:
            if idx < len(eval_dataset):
                item = eval_dataset[idx]
                self.raw_examples[idx] = {
                    "input_ids": item["input_ids"],
                    "attention_mask": item["attention_mask"],
                    "labels": item["labels"],
                }

    def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
        """Track predictions on specific examples after evaluation."""
        if model is None or state.epoch % self.log_every != 0:
            return

        model.eval()

        tracked_data = []
        for idx in self.tracked_indices:
            if idx in self.raw_examples:
                example = self.raw_examples[idx]

                # Get model predictions
                import torch

                with torch.no_grad():
                    input_ids = torch.tensor([example["input_ids"]]).to(model.device)
                    attention_mask = torch.tensor([example["attention_mask"]]).to(model.device)
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits.cpu().numpy()

                    # Convert to probabilities
                    probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
                    pred_label = np.argmax(probs, axis=1)[0]
                    confidence = np.max(probs, axis=1)[0]

                    # Store prediction history
                    self.history[idx].append(
                        {
                            "epoch": float(state.epoch),
                            "pred_label": int(pred_label),
                            "true_label": int(example["labels"]),
                            "confidence": float(confidence),
                            "match_prob": float(probs[0][1]),
                            "mismatch_prob": float(probs[0][0]),
                        }
                    )

                    # Decode input tokens for display
                    decoded_tokens = self.tokenizer.decode(example["input_ids"], skip_special_tokens=True)

                    tracked_data.append(
                        [
                            int(idx),
                            float(state.epoch),
                            decoded_tokens,
                            "Match" if example["labels"] == 1 else "Mismatch",
                            "Match" if pred_label == 1 else "Mismatch",
                            float(confidence),
                            1 if pred_label == example["labels"] else 0,
                        ]
                    )

        # Log to wandb
        if tracked_data:
            tracked_examples_table = wandb.Table(
                columns=["Index", "Epoch", "Text", "True Label", "Predicted Label", "Confidence", "Correct"], data=tracked_data
            )

            wandb.log({f"tracked_examples/epoch_{state.epoch}": tracked_examples_table})

    def on_train_end(self, args, state, control, **kwargs):
        """Log the prediction history of tracked examples at the end of training."""
        # Create a table showing how predictions evolved
        for idx in self.tracked_indices:
            if idx in self.history and len(self.history[idx]) > 0:
                history_data = []

                for entry in self.history[idx]:
                    history_data.append(
                        [
                            entry["epoch"],
                            "Match" if entry["true_label"] == 1 else "Mismatch",
                            "Match" if entry["pred_label"] == 1 else "Mismatch",
                            entry["confidence"],
                            entry["match_prob"],
                            entry["mismatch_prob"],
                            1 if entry["pred_label"] == entry["true_label"] else 0,
                        ]
                    )

                if history_data:
                    history_table = wandb.Table(
                        columns=["Epoch", "True Label", "Predicted Label", "Confidence", "Match Prob", "Mismatch Prob", "Correct"], data=history_data
                    )

                    wandb.log({f"prediction_history/example_{idx}": history_table})


class ModelAnalysisCallback(TrainerCallback):
    """Callback to analyze model architecture and parameters."""

    def __init__(self, model_name, tokenizer):
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.analyzed = False

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """Log model architecture and parameter stats at the beginning of training."""
        if model is None or self.analyzed:
            return

        # Log model card
        model_card = f"""
        # Model Card: {self.model_name}

        ## Base Model
        {self.model_name}

        ## Task
        Binary classification for check amount verification

        ## Architecture
        - Model type: RoBERTa-based classifier
        - Sequence classification head
        - Vocabulary size: {self.tokenizer.vocab_size}
        - Max sequence length: {self.tokenizer.model_max_length}
        """

        wandb.log({"model/model_card": model_card})

        # Parameter statistics
        param_stats = []
        param_sizes = {}

        for name, param in model.named_parameters():
            layer_name = name.split(".")[0] if "." in name else name
            param_sizes[layer_name] = param_sizes.get(layer_name, 0) + param.numel()

            param_stats.append(
                {
                    "name": name,
                    "shape": str(list(param.shape)),
                    "size": param.numel(),
                    "mean": float(param.detach().mean()),
                    "std": float(param.detach().std()),
                    "min": float(param.detach().min()),
                    "max": float(param.detach().max()),
                    "trainable": param.requires_grad,
                }
            )

        # Log parameter statistics
        param_stats_table = wandb.Table(columns=["Name", "Shape", "Size", "Mean", "Std", "Min", "Max", "Trainable"])

        for stat in param_stats:
            param_stats_table.add_data(
                stat["name"], stat["shape"], stat["size"], stat["mean"], stat["std"], stat["min"], stat["max"], stat["trainable"]
            )

        wandb.log({"model/parameter_stats": param_stats_table})

        # Log parameter size distribution
        param_sizes_table = wandb.Table(columns=["Layer", "Size"])

        for layer, size in param_sizes.items():
            param_sizes_table.add_data(layer, size)

        wandb.log({"model/parameter_sizes": param_sizes_table})

        # Mark as analyzed to avoid duplicate logging
        self.analyzed = True


class LearningDynamicsCallback(TrainerCallback):
    """Callback to track learning dynamics during training."""

    def __init__(self, log_every=100):
        self.log_every = log_every
        self.learning_metrics = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Track learning metrics during training."""
        if logs is None or state.global_step % self.log_every != 0:
            return

        # Extract metrics from logs
        metrics_dict = {}

        # Training metrics
        if "loss" in logs:
            metrics_dict["train/loss"] = logs["loss"]

        if "learning_rate" in logs:
            metrics_dict["train/learning_rate"] = logs["learning_rate"]

        # Evaluation metrics
        for key in logs:
            if key.startswith("eval_"):
                metric_name = key.replace("eval_", "eval/")
                metrics_dict[metric_name] = logs[key]

        # Add step information
        metrics_dict["global_step"] = state.global_step
        metrics_dict["epoch"] = state.epoch

        # Store metrics
        self.learning_metrics.append(metrics_dict)

        # Log to wandb
        if metrics_dict:
            wandb.log(metrics_dict)

    def on_train_end(self, args, state, control, **kwargs):
        """Log learning dynamics summary at the end of training."""
        if not self.learning_metrics:
            return

        # Create a summary of learning dynamics
        dynamics_data = []

        for metrics in self.learning_metrics:
            row = []

            # Extract common fields
            if "global_step" in metrics:
                row.append(metrics["global_step"])
            else:
                row.append(None)

            if "epoch" in metrics:
                row.append(metrics["epoch"])
            else:
                row.append(None)

            if "train/loss" in metrics:
                row.append(metrics["train/loss"])
            else:
                row.append(None)

            if "eval/loss" in metrics:
                row.append(metrics["eval/loss"])
            else:
                row.append(None)

            if "eval/accuracy" in metrics:
                row.append(metrics["eval/accuracy"])
            else:
                row.append(None)

            if "eval/f1" in metrics:
                row.append(metrics["eval/f1"])
            else:
                row.append(None)

            dynamics_data.append(row)

        # Create a table
        if dynamics_data:
            dynamics_table = wandb.Table(columns=["Step", "Epoch", "Train Loss", "Eval Loss", "Eval Accuracy", "Eval F1"], data=dynamics_data)

            wandb.log({"training/learning_dynamics": dynamics_table})


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

    # Log script version and execution time
    wandb.log(
        {
            "metadata/script_version": "2.0.0",
            "metadata/execution_time": wandb.util.generate_id(),
        }
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

    # Log dataset statistics
    wandb.log(
        {
            "dataset/train_size": len(train_dataset),
            "dataset/eval_size": len(eval_dataset),
            "dataset/test_size": len(test_dataset),
        }
    )

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

    wandb.log({"dataset/class_distribution": class_dist_table})

    # Initialize DataCollatorWithPadding if dynamic padding is enabled
    data_collator = DataCollatorWithPadding(tokenizer)

    # Initialize model
    logger.info("Initializing model...")
    model = AutoModelForSequenceClassification.from_pretrained(model_args.model_name_or_path, num_labels=2)

    # Select examples to track throughout training
    if training_args.track_learning_dynamics:
        track_indices = list(range(min(data_args.track_sample_count, len(eval_dataset))))
        logger.info(f"Tracking {len(track_indices)} examples throughout training")
    else:
        track_indices = []

    # Initialize trainer with additional callbacks
    callbacks = [
        EarlyStoppingCallback(early_stopping_patience=training_args.early_stopping_patience),
        GradientTrackingCallback(log_every=training_args.logging_steps),
    ]

    # Add learning dynamics callback if enabled
    if training_args.track_learning_dynamics:
        callbacks.append(LearningDynamicsCallback(log_every=training_args.logging_steps))

    # Add example tracking callback if enabled
    if track_indices:
        callbacks.append(ExampleTrackingCallback(eval_dataset, tokenizer, track_indices))

    # Add model analysis callback if enabled
    if training_args.log_model_architecture:
        callbacks.append(ModelAnalysisCallback(model_args.model_name_or_path, tokenizer))

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
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

    # Log model as an artifact
    model_artifact = wandb.Artifact(name=f"model-{wandb.run.id}", type="model", description="Trained DistilRoBERTa check amount verifier")

    model_artifact.add_dir(training_args.output_dir)
    wandb.log_artifact(model_artifact)

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
    }

    # Log model summary
    wandb.summary.update(model_summary)

    # Close wandb
    wandb.finish()


if __name__ == "__main__":
    main()
