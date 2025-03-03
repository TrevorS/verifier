"""
Evaluation module for assessing model performance and analyzing errors.
"""

import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

import config
from src.model import generate_text, load_model


# Custom JSON encoder to handle special values like Infinity and NaN
class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles special values like Infinity and NaN."""

    def default(self, obj):
        if isinstance(obj, float):
            if np.isnan(obj):
                return None  # Replace NaN with null in JSON
            elif np.isinf(obj):
                if obj > 0:
                    return "Infinity"  # Replace positive infinity with string "Infinity"
                else:
                    return "-Infinity"  # Replace negative infinity with string "-Infinity"
        return super().default(obj)


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
    model, tokenizer, metadata = load_model(model_path)

    # Load test data
    with open(test_data_path, "r") as f:
        test_examples = [json.loads(line) for line in f]

    # Evaluate each example
    results = []
    for example in tqdm(test_examples, desc="Evaluating"):
        input_text = example["input"]
        expected_output = example["target"] if "target" in example else example["output"]
        expected_amount = example.get("amount", None)

        # Generate prediction
        prediction = generate_text(model, tokenizer, input_text)

        # Calculate metrics
        result = analyze_prediction(input_text, expected_output, prediction, expected_amount)
        results.append(result)

    # Calculate overall metrics
    metrics = calculate_metrics(results)

    # Perform error analysis
    error_analysis = analyze_errors(results)
    metrics.update(error_analysis)

    # Save results and create visualizations
    save_evaluation_results(results, metrics, output_dir)
    create_visualizations(results, output_dir)

    # Generate comprehensive report
    generate_evaluation_report(results, metrics, output_dir)

    return metrics


def analyze_prediction(input_text, expected_output, prediction, expected_amount=None):
    """
    Analyze a single prediction against the expected output.

    Args:
        input_text (str): Input text
        expected_output (str): Expected output (pipe-delimited numeric amount)
        prediction (str): Model prediction (pipe-delimited numeric amount)
        expected_amount (float, optional): Expected amount if available

    Returns:
        dict: Dictionary with analysis results
    """
    # Calculate exact match
    exact_match = prediction.strip() == expected_output.strip()

    # Parse amounts
    try:
        expected_amount = float(expected_output)
        predicted_amount = float(prediction)
        amount_diff = abs(expected_amount - predicted_amount) if expected_amount is not None else None
        relative_diff = (amount_diff / expected_amount) if expected_amount != 0 else float("inf")
    except (ValueError, TypeError):
        amount_diff = None
        relative_diff = None

    # Determine error type
    if exact_match:
        error_type = "none"
    elif amount_diff == 0:
        error_type = "formatting"
    elif amount_diff is not None:
        error_type = "value_error"
    else:
        error_type = "structure"

    return {
        "input": input_text,
        "expected_output": expected_output,
        "prediction": prediction,
        "exact_match": exact_match,
        "amount_diff": amount_diff,
        "relative_diff": relative_diff,
        "error_type": error_type,
    }


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

    # Calculate amount differences
    amount_diffs = [result["amount_diff"] for result in results if result["amount_diff"] is not None]
    mean_amount_diff = np.mean(amount_diffs) if amount_diffs else "Infinity"  # Use string instead of float("inf")
    median_amount_diff = np.median(amount_diffs) if amount_diffs else "Infinity"  # Use string instead of float("inf")

    # Calculate relative differences
    relative_diffs = [result["relative_diff"] for result in results if result["relative_diff"] is not None]
    mean_relative_diff = np.mean(relative_diffs) if relative_diffs else "Infinity"  # Use string instead of float("inf")

    # Count examples with amount difference <= 0.01
    correct_amounts = sum(result["amount_diff"] <= 0.01 if result["amount_diff"] is not None else False for result in results)
    amount_accuracy = correct_amounts / total_examples if total_examples > 0 else 0

    # Error type counts
    error_types = [result["error_type"] for result in results]
    error_counts = {
        "none": error_types.count("none"),
        "formatting": error_types.count("formatting"),
        "value_error": error_types.count("value_error"),
        "structure": error_types.count("structure"),
    }

    return {
        "total_examples": total_examples,
        "exact_match_accuracy": exact_match_accuracy,
        "mean_amount_diff": mean_amount_diff,
        "median_amount_diff": median_amount_diff,
        "mean_relative_diff": mean_relative_diff,
        "amount_accuracy": amount_accuracy,
        "error_counts": error_counts,
    }


def analyze_errors(results):
    """
    Perform detailed error analysis on the results.

    Args:
        results (list): List of evaluation results

    Returns:
        dict: Dictionary with error analysis
    """
    # Group errors by type
    errors_by_type = defaultdict(list)
    for result in results:
        if result["error_type"] != "none":
            errors_by_type[result["error_type"]].append(result)

    # Analyze amount ranges
    amount_ranges = [
        (0, 1),
        (1, 10),
        (10, 100),
        (100, 1000),
        (1000, float("inf")),
    ]

    range_stats = {}
    for low, high in amount_ranges:
        range_name = f"{low}-{high if high != float('inf') else 'inf'}"
        examples_in_range = [r for r in results if r["expected_amount"] is not None and low <= r["expected_amount"] < high]

        if examples_in_range:
            correct = sum(1 for r in examples_in_range if r["exact_match"])
            accuracy = correct / len(examples_in_range)

            # Get valid amount differences
            valid_diffs = [r["amount_diff"] for r in examples_in_range if r["amount_diff"] is not None]
            if valid_diffs:
                avg_diff = np.mean(valid_diffs)
            else:
                avg_diff = None  # Use None instead of NaN

            range_stats[range_name] = {
                "count": len(examples_in_range),
                "accuracy": accuracy,
                "avg_diff": avg_diff,
            }

    # Find patterns in errors
    common_error_patterns = find_error_patterns(results)

    return {
        "errors_by_type_count": {k: len(v) for k, v in errors_by_type.items()},
        "range_stats": range_stats,
        "common_error_patterns": common_error_patterns,
    }


def find_error_patterns(results):
    """
    Find common patterns in errors.

    Args:
        results (list): List of evaluation results

    Returns:
        dict: Dictionary with error patterns
    """
    patterns = {}

    # Check for common issues with specific number forms
    number_forms = {
        "cents_only": [r for r in results if r["expected_amount"] < 1 and r["expected_amount"] > 0],
        "whole_dollars": [r for r in results if r["expected_amount"] % 1 == 0],
        "dollars_and_cents": [r for r in results if r["expected_amount"] % 1 != 0 and r["expected_amount"] >= 1],
        "thousands": [r for r in results if r["expected_amount"] >= 1000],
    }

    for form_name, form_results in number_forms.items():
        if form_results:
            correct = sum(1 for r in form_results if r["exact_match"])
            accuracy = correct / len(form_results) if form_results else 0
            patterns[form_name] = {
                "count": len(form_results),
                "accuracy": accuracy,
            }

    # Check for misspellings in input
    has_misspelling = []
    for r in results:
        if "hunred" in r["input"] or "thousnd" in r["input"] or "cnts" in r["input"] or "dolars" in r["input"]:
            has_misspelling.append(r)

    if has_misspelling:
        correct = sum(1 for r in has_misspelling if r["exact_match"])
        accuracy = correct / len(has_misspelling)
        patterns["misspellings"] = {
            "count": len(has_misspelling),
            "accuracy": accuracy,
        }

    return patterns


def analyze_specific_example(result):
    """
    Analyze a specific example in detail.

    Args:
        result (dict): Result dictionary for a single example

    Returns:
        dict: Detailed analysis
    """
    analysis = {
        "input": result["input"],
        "expected": result["expected_output"],
        "prediction": result["prediction"],
        "is_exact_match": result["exact_match"],
        "error_type": result["error_type"],
    }

    # Highlight differences
    if not result["exact_match"]:
        analysis["expected_amount"] = result["expected_amount"]
        analysis["predicted_amount"] = result["predicted_amount"]

        if result["amount_diff"] is not None:
            analysis["absolute_diff"] = result["amount_diff"]
            analysis["relative_diff"] = result["relative_diff"] if result["relative_diff"] is not None else "N/A"

    return analysis


def create_visualizations(results, output_dir):
    """
    Create visualizations of the evaluation results.

    Args:
        results (list): List of evaluation results
        output_dir (str or Path): Directory to save visualizations
    """
    # Ensure the visualization directory exists
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    # Get valid results with amounts
    valid_results = [r for r in results if r["amount_diff"] is not None]

    if not valid_results:
        # Create a simple error distribution chart even when no valid amount predictions
        plt.figure(figsize=(10, 8))
        error_types = [r["error_type"] for r in results]
        error_counts = {
            "No Error": error_types.count("none"),
            "Formatting": error_types.count("formatting"),
            "Value Error": error_types.count("value_error"),
            "Structure": error_types.count("structure"),
        }

        plt.pie(
            error_counts.values(),
            labels=error_counts.keys(),
            autopct="%1.1f%%",
            startangle=90,
            shadow=True,
        )
        plt.axis("equal")
        plt.title("Distribution of Error Types")
        plt.savefig(os.path.join(viz_dir, "error_types.png"))
        plt.close()

        # Save empty placeholder images for other charts with error messages
        for chart_name, title in [
            ("accuracy_by_range.png", "Accuracy by Amount Range"),
            ("error_distribution.png", "Distribution of Prediction Errors"),
            ("expected_vs_predicted.png", "Expected vs Predicted Amounts"),
        ]:
            plt.figure(figsize=(10, 6))
            plt.text(
                0.5,
                0.5,
                "No valid predictions available for visualization",
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=14,
            )
            plt.title(title)
            plt.axis("off")
            plt.savefig(os.path.join(viz_dir, chart_name))
            plt.close()

        return  # Skip the rest of the visualizations

    # 1. Plot accuracy by amount range
    amount_ranges = [
        (0, 1),
        (1, 10),
        (10, 100),
        (100, 1000),
        (1000, float("inf")),
    ]

    range_names = []
    accuracies = []

    for low, high in amount_ranges:
        range_name = f"{low}-{high if high != float('inf') else 'inf'}"
        examples_in_range = [r for r in results if r["expected_amount"] is not None and low <= r["expected_amount"] < high]

        if examples_in_range:
            correct = sum(1 for r in examples_in_range if r["exact_match"])
            accuracy = correct / len(examples_in_range)

            range_names.append(range_name)
            accuracies.append(accuracy)

    plt.figure(figsize=(10, 6))
    plt.bar(range_names, accuracies)
    plt.xlabel("Amount Range")
    plt.ylabel("Accuracy")
    plt.title("Accuracy by Amount Range")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.savefig(os.path.join(viz_dir, "accuracy_by_range.png"))
    plt.close()

    # 2. Distribution of numeric errors
    amount_diffs = [r["amount_diff"] for r in valid_results]

    plt.figure(figsize=(10, 6))
    plt.hist(amount_diffs, bins=20, alpha=0.7)
    plt.xlabel("Absolute Error")
    plt.ylabel("Frequency")
    plt.title("Distribution of Prediction Errors")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.savefig(os.path.join(viz_dir, "error_distribution.png"))
    plt.close()

    # 3. Error types pie chart
    error_types = [r["error_type"] for r in results]
    error_counts = {
        "No Error": error_types.count("none"),
        "Formatting": error_types.count("formatting"),
        "Value Error": error_types.count("value_error"),
        "Structure": error_types.count("structure"),
    }

    plt.figure(figsize=(10, 8))
    plt.pie(
        error_counts.values(),
        labels=error_counts.keys(),
        autopct="%1.1f%%",
        startangle=90,
        shadow=True,
    )
    plt.axis("equal")
    plt.title("Distribution of Error Types")
    plt.savefig(os.path.join(viz_dir, "error_types.png"))
    plt.close()

    # 4. Scatter plot of expected vs predicted amounts
    expected_amounts = [r["expected_amount"] for r in valid_results]
    predicted_amounts = [r["predicted_amount"] for r in valid_results]

    plt.figure(figsize=(10, 8))
    plt.scatter(expected_amounts, predicted_amounts, alpha=0.6)

    # Add a perfect prediction line
    max_val = max(max(expected_amounts), max(predicted_amounts))
    min_val = min(min(expected_amounts), min(predicted_amounts))
    plt.plot([min_val, max_val], [min_val, max_val], "r--")

    plt.xlabel("Expected Amount")
    plt.ylabel("Predicted Amount")
    plt.title("Expected vs Predicted Amounts")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig(os.path.join(viz_dir, "expected_vs_predicted.png"))
    plt.close()


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

    # Save metrics - use custom JSON encoder to handle special values
    metrics_path = os.path.join(output_dir, "evaluation_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, cls=CustomJSONEncoder)

    # Save example predictions
    examples_path = os.path.join(output_dir, "example_predictions.json")
    sample_results = []

    # Include both correct and incorrect examples
    correct_examples = [r for r in results if r["exact_match"]][:10]
    incorrect_examples = [r for r in results if not r["exact_match"]][:20]

    for result in correct_examples + incorrect_examples:
        sample_results.append(analyze_specific_example(result))

    with open(examples_path, "w") as f:
        json.dump(sample_results, f, indent=2, cls=CustomJSONEncoder)

    return results_path, metrics_path, examples_path


def generate_evaluation_report(results, metrics, output_dir):
    """
    Generate a detailed evaluation report in markdown format.

    Args:
        results (list): List of evaluation results
        metrics (dict): Dictionary of metrics
        output_dir (str or Path): Directory to save the report

    Returns:
        str: Path to the saved report
    """
    report_path = os.path.join(output_dir, "evaluation_report.md")

    # Helper function to format float values safely
    def format_value(value, precision=4):
        if isinstance(value, float):
            if np.isnan(value):
                return "N/A"
            elif np.isinf(value):
                return "∞" if value > 0 else "-∞"
            else:
                return f"{value:.{precision}f}"
        return str(value)

    with open(report_path, "w") as f:
        # Title
        f.write("# Model Evaluation Report\n\n")

        # Overall metrics
        f.write("## Overall Metrics\n\n")
        f.write(f"- **Total examples**: {metrics['total_examples']}\n")
        f.write(f"- **Exact match accuracy**: {format_value(metrics['exact_match_accuracy'])}\n")
        f.write(f"- **Amount accuracy (diff ≤ 0.01)**: {format_value(metrics['amount_accuracy'])}\n")
        f.write(f"- **Mean amount difference**: {format_value(metrics['mean_amount_diff'])}\n")
        f.write(f"- **Median amount difference**: {format_value(metrics['median_amount_diff'])}\n")
        f.write(f"- **Mean relative difference**: {format_value(metrics['mean_relative_diff'])}\n\n")

        # Error breakdown
        f.write("## Error Breakdown\n\n")
        f.write("| Error Type | Count | Percentage |\n")
        f.write("|------------|-------|------------|\n")
        for error_type, count in metrics["error_counts"].items():
            percentage = count / metrics["total_examples"] * 100
            f.write(f"| {error_type} | {count} | {percentage:.2f}% |\n")

        f.write("\n")

        # Performance by amount range
        if "range_stats" in metrics:
            f.write("## Performance by Amount Range\n\n")
            f.write("| Amount Range | Count | Accuracy | Average Error |\n")
            f.write("|--------------|-------|----------|---------------|\n")

            for range_name, stats in metrics["range_stats"].items():
                f.write(f"| ${range_name} | {stats['count']} | {format_value(stats['accuracy'])} | {format_value(stats['avg_diff'])} |\n")

            f.write("\n")

        # Common error patterns
        if "common_error_patterns" in metrics:
            f.write("## Common Error Patterns\n\n")

            for pattern_name, stats in metrics["common_error_patterns"].items():
                f.write(f"### {pattern_name.replace('_', ' ').title()}\n")
                f.write(f"- **Count**: {stats['count']}\n")
                f.write(f"- **Accuracy**: {format_value(stats['accuracy'])}\n\n")

        # Sample predictions
        f.write("## Sample Predictions\n\n")

        # Show a few correct predictions
        correct_examples = [r for r in results if r["exact_match"]][:3]
        if correct_examples:
            f.write("### Correct Predictions\n\n")

            for i, example in enumerate(correct_examples, 1):
                f.write(f"**Example {i}:**\n")
                f.write(f'- Input: "{example["input"]}"\n')
                f.write(f"- Expected: `{example['expected_output']}`\n")
                f.write(f"- Predicted: `{example['prediction']}`\n\n")

        # Show a few incorrect predictions
        incorrect_examples = [r for r in results if not r["exact_match"]][:5]
        if incorrect_examples:
            f.write("### Incorrect Predictions\n\n")

            for i, example in enumerate(incorrect_examples, 1):
                f.write(f"**Example {i}:**\n")
                f.write(f'- Input: "{example["input"]}"\n')
                f.write(f"- Expected: `{example['expected_output']}`\n")
                f.write(f"- Predicted: `{example['prediction']}`\n")

                if example["amount_diff"] is not None:
                    f.write(f"- Expected amount: {example['expected_amount']}\n")
                    f.write(f"- Predicted amount: {example['predicted_amount']}\n")
                    f.write(f"- Difference: {format_value(example['amount_diff'])}\n")

                f.write(f"- Error type: {example['error_type']}\n\n")

        # Visualizations reference
        f.write("## Visualizations\n\n")
        f.write("The following visualizations were generated:\n\n")
        f.write("1. Accuracy by amount range: `visualizations/accuracy_by_range.png`\n")
        f.write("2. Distribution of prediction errors: `visualizations/error_distribution.png`\n")
        f.write("3. Distribution of error types: `visualizations/error_types.png`\n")
        f.write("4. Expected vs predicted amounts: `visualizations/expected_vs_predicted.png`\n\n")

        # Recommendations
        f.write("## Recommendations for Improvement\n\n")

        # Add some generic recommendations
        f.write("Based on the error analysis, here are some recommendations for improvement:\n\n")

        # Check for specific patterns and provide targeted recommendations
        if metrics.get("common_error_patterns", {}).get("cents_only", {}).get("accuracy", 1.0) < 0.9:
            f.write("1. **Improve handling of cents-only amounts**: The model struggles with inputs that only mention cents.\n")

        if metrics.get("common_error_patterns", {}).get("thousands", {}).get("accuracy", 1.0) < 0.9:
            f.write("2. **Focus on large numbers**: Improve handling of amounts in the thousands.\n")

        if metrics.get("common_error_patterns", {}).get("misspellings", {}).get("accuracy", 1.0) < 0.9:
            f.write("3. **Enhance robustness to spelling errors**: The model's performance drops with misspelled inputs.\n")

        f.write("4. **Additional training data**: Consider generating more diverse training examples.\n")
        f.write("5. **Model size**: Experiment with larger model sizes if resources permit.\n")

    return report_path


def get_error_examples(results, error_type, n=5):
    """
    Get examples of a specific error type.

    Args:
        results (list): List of evaluation results
        error_type (str): Type of error to get examples for
        n (int): Number of examples to return

    Returns:
        list: List of examples
    """
    error_examples = [r for r in results if r["error_type"] == error_type]
    return error_examples[:n]
