"""
Evaluation module for assessing model performance.
"""

import json
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

import config
from src.model import generate_text, load_model


def evaluate_model(model_path, test_data_path=None, output_dir=None):
    """
    Evaluate the model on test data.

    Args:
        model_path (str or Path): Path to the trained model
        test_data_path (str or Path): Path to the test data
        output_dir (str or Path): Directory to save evaluation results

    Returns:
        dict: Dictionary of evaluation metrics
    """
    # Use default values from config if not provided
    if test_data_path is None:
        test_data_path = config.TEST_DATA_PATH

    if output_dir is None:
        output_dir = config.ROOT_DIR / "eval_results"

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the model and tokenizer
    model, tokenizer = load_model(model_path)

    # Load test data
    with open(test_data_path, "r") as f:
        test_examples = [json.loads(line) for line in f]

    # Evaluate each example
    results = []
    for example in tqdm(test_examples, desc="Evaluating"):
        input_text = example["input"]
        expected_output = example["output"]

        # Generate prediction
        prediction = generate_text(model, tokenizer, input_text)

        # Parse JSON
        try:
            expected_json = json.loads(expected_output)
            predicted_json = json.loads(prediction)
            json_valid = True

            # Extract amounts
            if "amount" in expected_json and "amount" in predicted_json:
                expected_amount = expected_json["amount"]
                predicted_amount = predicted_json["amount"]
                amount_diff = abs(expected_amount - predicted_amount)
            else:
                expected_amount = None
                predicted_amount = None
                amount_diff = None

        except (json.JSONDecodeError, ValueError, TypeError, KeyError):
            json_valid = False
            expected_amount = None
            predicted_amount = None
            amount_diff = None

        # Calculate exact match
        exact_match = prediction == expected_output

        # Store result
        results.append(
            {
                "input": input_text,
                "expected_output": expected_output,
                "prediction": prediction,
                "exact_match": exact_match,
                "json_valid": json_valid,
                "expected_amount": expected_amount,
                "predicted_amount": predicted_amount,
                "amount_diff": amount_diff,
            }
        )

    # Calculate overall metrics
    metrics = calculate_metrics(results)

    # Save results
    save_evaluation_results(results, metrics, output_dir)

    return metrics


def calculate_metrics(results):
    """
    Calculate evaluation metrics from results.

    Args:
        results (list): List of evaluation results

    Returns:
        dict: Dictionary of metrics
    """
    # Count total examples
    total_examples = len(results)

    # Count exact matches
    exact_matches = sum(result["exact_match"] for result in results)
    exact_match_accuracy = exact_matches / total_examples if total_examples > 0 else 0

    # Count valid JSON
    valid_json = sum(result["json_valid"] for result in results)
    json_validity = valid_json / total_examples if total_examples > 0 else 0

    # Calculate amount differences
    amount_diffs = [
        result["amount_diff"] for result in results if result["amount_diff"] is not None
    ]
    mean_amount_diff = np.mean(amount_diffs) if amount_diffs else float("inf")
    median_amount_diff = np.median(amount_diffs) if amount_diffs else float("inf")

    # Count examples with amount difference <= 0.01
    correct_amounts = sum(
        result["amount_diff"] <= 0.01 if result["amount_diff"] is not None else False
        for result in results
    )
    amount_accuracy = correct_amounts / total_examples if total_examples > 0 else 0

    return {
        "total_examples": total_examples,
        "exact_match_accuracy": exact_match_accuracy,
        "json_validity": json_validity,
        "mean_amount_diff": mean_amount_diff,
        "median_amount_diff": median_amount_diff,
        "amount_accuracy": amount_accuracy,
    }


def save_evaluation_results(results, metrics, output_dir):
    """
    Save evaluation results and metrics.

    Args:
        results (list): List of evaluation results
        metrics (dict): Dictionary of metrics
        output_dir (str or Path): Directory to save results

    Returns:
        tuple: Paths to saved result files
    """
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Save results
    results_path = os.path.join(output_dir, "evaluation_results.csv")
    results_df.to_csv(results_path, index=False)

    # Save metrics
    metrics_path = os.path.join(output_dir, "evaluation_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Save example predictions
    examples_path = os.path.join(output_dir, "example_predictions.json")
    examples = [
        {
            "input": result["input"],
            "expected_output": result["expected_output"],
            "prediction": result["prediction"],
            "exact_match": result["exact_match"],
            "json_valid": result["json_valid"],
            "amount_diff": result["amount_diff"],
        }
        for result in results[:20]  # Save first 20 examples
    ]
    with open(examples_path, "w") as f:
        json.dump(examples, f, indent=2)

    return results_path, metrics_path, examples_path
