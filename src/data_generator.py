"""
Data generation module for creating synthetic training data.
"""

import json
import os
import random
import string
from pathlib import Path

import inflect
import numpy as np

from src.utils import format_json, normalize_text, number_to_words


def generate_random_amount(min_amount=0.01, max_amount=1000000.00):
    """
    Generate a random monetary amount within the specified range.

    Args:
        min_amount (float): Minimum amount (inclusive)
        max_amount (float): Maximum amount (inclusive)

    Returns:
        float: A random monetary amount
    """
    # Generate amount using log-uniform distribution to ensure good coverage
    # across orders of magnitude (e.g., cents, dollars, thousands, etc.)
    if min_amount <= 0:
        min_amount = 0.01  # Ensure positive values for log distribution

    log_min = np.log10(min_amount)
    log_max = np.log10(max_amount)

    # Generate a random value in log space
    log_amount = random.uniform(log_min, log_max)

    # Convert back to linear space
    amount = 10**log_amount

    # Round to 2 decimal places (cents)
    amount = round(amount * 100) / 100

    return amount


def generate_stratified_amounts(num_examples):
    """
    Generate a stratified sample of monetary amounts across different ranges.

    Args:
        num_examples (int): Number of examples to generate

    Returns:
        list: List of monetary amounts with good distribution
    """
    amounts = []

    # Define ranges for stratification
    ranges = [
        (0.01, 0.99),  # Cents only
        (1.00, 9.99),  # Single-digit dollars
        (10.00, 99.99),  # Double-digit dollars
        (100.00, 999.99),  # Triple-digit dollars
        (1000.00, 9999.99),  # Thousands
        (10000.00, 99999.99),  # Tens of thousands
        (100000.00, 1000000.00),  # Hundreds of thousands to million
    ]

    # Assign more weight to common ranges (cents, single, double, triple-digit dollars)
    weights = [0.15, 0.20, 0.20, 0.15, 0.15, 0.10, 0.05]

    # Determine number of examples per range
    counts = [int(num_examples * w) for w in weights]

    # Adjust to ensure we get exactly num_examples
    remainder = num_examples - sum(counts)
    counts[0] += remainder

    # Generate amounts for each range
    for (min_val, max_val), count in zip(ranges, counts):
        for _ in range(count):
            amounts.append(generate_random_amount(min_val, max_val))

    # Shuffle the amounts
    random.shuffle(amounts)

    return amounts


def amount_to_verbal_expression(amount, variation_type=None):
    """
    Convert a numerical monetary amount to a verbal expression with variations.

    Args:
        amount (float): Monetary amount
        variation_type (str, optional): Type of variation to generate
            ('standard', 'with_cents', 'cents_only', 'no_and', 'dollars_only')
            If None, a random variation will be chosen.

    Returns:
        str: Verbal expression of the monetary amount
    """
    dollars = int(amount)
    cents = int(round((amount - dollars) * 100))

    if variation_type is None:
        # Choose a random variation based on the amount
        # Only include 'cents_only' option when there are only cents (no dollars)
        if dollars == 0 and cents > 0:
            variation_type = random.choice(["cents_only", "standard"])
        # Only include 'dollars_only' option when there are only dollars (no cents)
        elif dollars > 0 and cents == 0:
            variation_type = random.choice(["dollars_only", "standard", "with_cents"])
        else:
            # Both dollars and cents
            variation_type = random.choice(["standard", "no_and"])

    if variation_type == "standard":
        # Standard format: "X dollars and Y cents" or just "X dollars" or just "Y cents"
        return number_to_words(amount, include_and=True)

    elif variation_type == "with_cents":
        # Explicitly mention zero cents: "X dollars and zero cents"
        if dollars > 0:
            dollar_words = number_to_words(dollars, include_and=False)
            if cents == 0:
                return f"{dollar_words} and zero cents"
            else:
                cent_words = number_to_words(cents / 100, include_and=False)
                cent_words = cent_words.replace("point ", "")
                return f"{dollar_words} and {cent_words}"
        else:
            # Just cents, no dollars
            return number_to_words(amount, include_and=True)

    elif variation_type == "cents_only":
        # Express only in cents: "X cents" (only for amounts < 1.00)
        if dollars == 0 and cents > 0:
            p = inflect.engine()
            cent_words = p.number_to_words(cents)
            return f"{cent_words} cents"
        else:
            # Fallback to standard for amounts >= 1.00
            return number_to_words(amount, include_and=True)

    elif variation_type == "no_and":
        # Omit the 'and': "X dollars Y cents"
        return number_to_words(amount, include_and=False)

    elif variation_type == "dollars_only":
        # Express only in dollars: "X dollars" (for whole dollar amounts)
        if cents == 0 and dollars > 0:
            return f"{number_to_words(dollars, include_and=False)} dollars"
        else:
            # Fallback to standard if there are cents
            return number_to_words(amount, include_and=True)

    # Fallback to standard format
    return number_to_words(amount, include_and=True)


def create_json_output(amount):
    """
    Create a JSON representation of a monetary amount.

    Args:
        amount (float): Monetary amount

    Returns:
        str: JSON string representation
    """
    return format_json(amount)


def apply_augmentation(text, dropout_prob=0.05, case_change_prob=0.2):
    """
    Apply text augmentation to the input text.

    Args:
        text (str): Input text
        dropout_prob (float): Probability of dropping a character
        case_change_prob (float): Probability of changing case (before normalization)

    Returns:
        str: Augmented text
    """
    # Apply case change before dropout (will be normalized later)
    if random.random() < case_change_prob:
        if random.random() < 0.5:
            # Uppercase a random word
            words = text.split()
            if words:
                idx = random.randint(0, len(words) - 1)
                words[idx] = words[idx].upper()
                text = " ".join(words)
        else:
            # Random capitalization
            text = "".join(c.upper() if random.random() < 0.3 else c for c in text)

    # Character dropout
    if random.random() < 0.3:  # Only apply dropout to 30% of samples
        chars = []
        for c in text:
            if c in string.whitespace or random.random() > dropout_prob:
                chars.append(c)
        text = "".join(chars)

    # Add typo (character substitution)
    if random.random() < 0.1:  # Apply to 10% of samples
        chars = list(text)
        if chars:
            idx = random.randint(0, len(chars) - 1)
            if chars[idx] not in string.whitespace:
                # Replace with a random character
                chars[idx] = random.choice(string.ascii_lowercase)
            text = "".join(chars)

    # Add extra whitespace
    if random.random() < 0.2:  # Apply to 20% of samples
        words = text.split()
        if words:
            idx = random.randint(0, len(words) - 1)
            words[idx] = "  " + words[idx] + "  "
            text = " ".join(words)

    # Normalize the text (lowercase and standardize whitespace)
    return normalize_text(text)


def generate_examples(num_examples):
    """
    Generate examples of verbal monetary expressions and their JSON representations.

    Args:
        num_examples (int): Number of examples to generate

    Returns:
        list: List of dictionaries with input and target fields
    """
    # Generate stratified amounts
    amounts = generate_stratified_amounts(num_examples)

    examples = []
    for amount in amounts:
        # Generate verbal expression (with random variation)
        verbal_expr = amount_to_verbal_expression(amount)

        # Apply augmentation to some examples
        if random.random() < 0.3:  # Apply augmentation to 30% of examples
            verbal_expr = apply_augmentation(verbal_expr)

        # Create JSON output
        json_output = create_json_output(amount)

        # Create example
        example = {
            "input": verbal_expr,
            "target": json_output,
            # Store original amount for reference
            "amount": float(format(amount, ".2f")),
        }

        examples.append(example)

    return examples


def generate_dataset(num_examples=10000, output_dir=None, train_ratio=0.8):
    """
    Generate a synthetic dataset of verbal monetary expressions and their JSON
    representations.

    Args:
        num_examples (int): Number of examples to generate
        output_dir (str or Path): Directory to save the generated data
        train_ratio (float): Ratio of examples to use for training (vs validation)

    Returns:
        tuple: Paths to the generated train and validation files
    """
    # Set up output directory
    if output_dir is None:
        # Default to 'data' directory in the project root
        output_dir = Path(__file__).parents[1] / "data"
    else:
        output_dir = Path(output_dir)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Calculate split sizes
    train_size = int(num_examples * train_ratio)
    # The validation size is implicitly num_examples - train_size

    # Generate examples
    examples = generate_examples(num_examples)

    # Shuffle examples
    random.shuffle(examples)

    # Split into train and validation sets
    train_examples = examples[:train_size]
    val_examples = examples[train_size:]

    # Save to files
    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"

    # Write train data
    with open(train_path, "w") as f:
        for example in train_examples:
            f.write(json.dumps(example) + "\n")

    # Write validation data
    with open(val_path, "w") as f:
        for example in val_examples:
            f.write(json.dumps(example) + "\n")

    print(f"Generated {len(train_examples)} training examples and {len(val_examples)} validation examples")
    print(f"Training data saved to: {train_path}")
    print(f"Validation data saved to: {val_path}")

    # Generate a sample of examples for inspection
    sample_size = min(10, num_examples)
    sample_examples = random.sample(examples, sample_size)

    print("\nSample examples:")
    for i, example in enumerate(sample_examples, 1):
        print(f"\nExample {i}:")
        print(f"Input:  {example['input']}")
        print(f"Target: {example['target']}")
        print(f"Amount: ${example['amount']:.2f}")

    return train_path, val_path


if __name__ == "__main__":
    # Generate a small sample dataset for testing
    num_examples = 1000
    train_path, val_path = generate_dataset(num_examples=num_examples)

    # Load and analyze the distribution of amounts
    amount_ranges = {
        "cents_only": 0,
        "single_digit": 0,
        "double_digit": 0,
        "triple_digit": 0,
        "thousands": 0,
        "tens_thousands": 0,
        "hundreds_thousands+": 0,
    }

    all_examples = []
    # Read both files
    for path in [train_path, val_path]:
        with open(path, "r") as f:
            for line in f:
                example = json.loads(line)
                all_examples.append(example)

    # Count examples in each range
    for example in all_examples:
        amount = example["amount"]
        if amount < 1:
            amount_ranges["cents_only"] += 1
        elif amount < 10:
            amount_ranges["single_digit"] += 1
        elif amount < 100:
            amount_ranges["double_digit"] += 1
        elif amount < 1000:
            amount_ranges["triple_digit"] += 1
        elif amount < 10000:
            amount_ranges["thousands"] += 1
        elif amount < 100000:
            amount_ranges["tens_thousands"] += 1
        else:
            amount_ranges["hundreds_thousands+"] += 1

    # Print distribution summary
    print("\nAmount range distribution:")
    for range_name, count in amount_ranges.items():
        percentage = (count / num_examples) * 100
        print(f"{range_name}: {count} examples ({percentage:.1f}%)")
