"""
Data generation module for creating synthetic training data.
"""

import json
import os
import random
from pathlib import Path


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
    examples = []
    for _ in range(num_examples):
        # Placeholder for data generation logic
        # This will be implemented in the future
        pass

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

    return train_path, val_path
