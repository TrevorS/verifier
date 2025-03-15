#!/usr/bin/env python3
"""
Main entry point for the monetary expressions to numeric amount converter.
"""

import argparse
import logging
import sys
from pathlib import Path

from src.data_generator import create_complete_dataset

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(Path(__file__).parent / "app.log"),
    ],
)
logger = logging.getLogger(__name__)


def generate_data(args):
    """Generate synthetic training data."""
    logger.info("Generating synthetic data...")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Number of examples: {args.num_examples}")

    dataset, _ = create_complete_dataset(
        num_examples=args.num_examples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=1.0 - args.train_ratio - args.val_ratio,
        output_dir=args.output_dir,
        seed=args.seed,
        augmentation_ratio=args.augmentation_ratio,
        hard_examples_ratio=args.hard_examples_ratio,
    )

    logger.info("Data generation completed.")
    logger.info(f"Dataset splits: {list(dataset.keys())}")
    logger.info(f"Train examples: {len(dataset['train'])}")
    logger.info(f"Validation examples: {len(dataset['validation'])}")
    logger.info(f"Test examples: {len(dataset['test'])}")
    logger.info(f"Files saved to: {args.output_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Monetary expressions to numeric amount converter")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for generated data")
    parser.add_argument("--num-examples", type=int, required=True, help="Number of examples to generate")
    parser.add_argument("--train-ratio", type=float, required=True, help="Training data ratio")
    parser.add_argument("--val-ratio", type=float, required=True, help="Validation data ratio")
    parser.add_argument("--seed", type=int, required=True, help="Random seed")
    parser.add_argument("--augmentation-ratio", type=float, required=True, help="Augmentation ratio")
    parser.add_argument("--hard-examples-ratio", type=float, required=True, help="Hard examples ratio")
    args = parser.parse_args()

    generate_data(args)


if __name__ == "__main__":
    main()
