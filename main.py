#!/usr/bin/env python3
"""
Main entry point for the monetary expressions to numeric amount converter.
"""

import argparse
import logging
import sys
from pathlib import Path

import config

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=getattr(logging, config.LOG_LEVEL),
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(Path(config.ROOT_DIR) / "app.log"),
    ],
)
logger = logging.getLogger(__name__)


def train(args):
    """Train the model."""
    logger.info("Starting training...")
    logger.info(f"Using model: {args.model_name}")

    # Determine data paths based on whether we're using a small subset for testing
    train_data_path = args.train_data_path
    val_data_path = args.val_data_path

    if args.test_run:
        logger.info("Running with a small subset of data for testing")
        # Import necessary modules
        import json

        from src.dataset import load_dataset

        # Create temp directory if it doesn't exist
        temp_dir = Path(config.ROOT_DIR) / "temp"
        temp_dir.mkdir(exist_ok=True)

        # Load original datasets
        train_dataset = load_dataset(train_data_path)["train"]
        val_dataset = load_dataset(val_data_path)["validation"]

        # Select a small subset (e.g., 100 examples for training, 20 for validation)
        train_subset = train_dataset.select(range(min(100, len(train_dataset))))
        val_subset = val_dataset.select(range(min(20, len(val_dataset))))

        # Define temporary file paths
        train_data_path = temp_dir / "train_subset.jsonl"
        val_data_path = temp_dir / "val_subset.jsonl"

        # Save subsets as JSONL
        with open(train_data_path, "w") as f:
            for item in train_subset:
                f.write(json.dumps(item) + "\n")

        with open(val_data_path, "w") as f:
            for item in val_subset:
                f.write(json.dumps(item) + "\n")

        logger.info(f"Created temporary training data: {train_data_path} ({len(train_subset)} examples)")
        logger.info(f"Created temporary validation data: {val_data_path} ({len(val_subset)} examples)")
    else:
        logger.info(f"Training data: {train_data_path}")
        logger.info(f"Validation data: {val_data_path}")

    # Import here to avoid loading modules unnecessarily
    from src.trainer import train_model

    best_model_path = train_model(
        model_name=args.model_name,
        train_data_path=train_data_path,
        val_data_path=val_data_path,
        output_dir=args.output_dir,
        wandb_logging=not args.no_wandb,
        early_stopping_patience=args.early_stopping_patience,
        quick_test=args.quick_test if hasattr(args, "quick_test") else False,
    )
    logger.info(f"Training completed. Best model saved at: {best_model_path}")

    return best_model_path


def evaluate(args):
    """Evaluate the model on test data."""
    logger.info("Starting evaluation...")
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Test data: {args.test_data_path}")

    # Import here to avoid loading modules unnecessarily
    from src.evaluation import evaluate_model

    evaluate_model(
        model_path=args.model_path,
        test_data_path=args.test_data_path,
        output_dir=args.output_dir,
    )
    logger.info("Evaluation completed.")


def infer(args):
    """Run inference with the model."""
    logger.info("Starting inference...")
    logger.info(f"Model path: {args.model_path}")

    # Import here to avoid loading modules unnecessarily
    from src.inference import demo_inference, inference_pipeline

    if args.demo:
        # Run the demo with example inputs
        logger.info("Running demo with example inputs")
        demo_inference(args.model_path)
        return

    if args.input_file:
        logger.info(f"Input file: {args.input_file}")
        with open(args.input_file, "r") as f:
            texts = [line.strip() for line in f if line.strip()]

        logger.info(f"Processing {len(texts)} inputs from file")
        # Process each input
        for text in texts:
            amount = inference_pipeline(text=text, model_path=args.model_path)

            print("\n" + "-" * 50)
            print(f"Input: {text}")
            print(f"Output: {amount}")

    else:
        # Process single input
        text = args.text
        logger.info(f"Processing input: {text}")

        amount = inference_pipeline(text=text, model_path=args.model_path)

        print("\n" + "-" * 50)
        print(f"Input: {text}")
        print(f"Output: {amount}")

    logger.info("Inference completed.")


def generate_data(args):
    """Generate synthetic training data."""
    logger.info("Generating synthetic data...")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Number of examples: {args.num_examples}")

    # Import here to avoid loading modules unnecessarily
    from src.data_generator import create_complete_dataset

    dataset, paths = create_complete_dataset(
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
    parser = argparse.ArgumentParser(description=("A sequence-to-sequence model that converts verbal monetary expressions to numeric amounts."))
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Train parser
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument(
        "--model-name",
        type=str,
        default=config.MODEL_NAME,
        help="Name of the pretrained model to fine-tune",
    )
    train_parser.add_argument(
        "--train-data-path",
        type=str,
        default=str(config.TRAIN_DATA_PATH),
        help="Path to the training data",
    )
    train_parser.add_argument(
        "--val-data-path",
        type=str,
        default=str(config.VAL_DATA_PATH),
        help="Path to the validation data",
    )
    train_parser.add_argument(
        "--output-dir",
        type=str,
        default=str(config.MODELS_DIR / "checkpoints"),
        help="Directory to save model checkpoints",
    )
    train_parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging",
    )
    train_parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=3,
        help="Number of evaluations with no improvement after which to stop training",
    )
    train_parser.add_argument(
        "--test-run",
        action="store_true",
        help="Run with a small subset of data to test the pipeline end-to-end",
    )
    train_parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Run a quick test with a small subset of data",
    )

    # Evaluate parser
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate the model")
    eval_parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the trained model",
    )
    eval_parser.add_argument(
        "--test-data-path",
        type=str,
        default=str(config.TEST_DATA_PATH),
        help="Path to the test data",
    )
    eval_parser.add_argument(
        "--output-dir",
        type=str,
        default=str(config.ROOT_DIR / "eval_results"),
        help="Directory to save evaluation results",
    )

    # Inference parser
    infer_parser = subparsers.add_parser("infer", help="Run inference")
    infer_parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the trained model",
    )
    infer_parser.add_argument(
        "--text",
        type=str,
        default="twenty-five dollars and ten cents",
        help="Text to infer from",
    )
    infer_parser.add_argument(
        "--input-file",
        type=str,
        help="Path to file containing input texts (one per line)",
    )
    infer_parser.add_argument(
        "--demo",
        action="store_true",
        help="Run the demo with example inputs",
    )

    # Data generation parser
    data_parser = subparsers.add_parser("generate-data", help="Generate synthetic data")
    data_parser.add_argument(
        "--num-examples",
        type=int,
        default=100000,
        help="Total number of examples to generate",
    )
    data_parser.add_argument(
        "--output-dir",
        type=str,
        default=str(config.DATA_DIR),
        help="Directory to save generated data",
    )
    data_parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Ratio of examples to use for training",
    )
    data_parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Ratio of examples to use for validation",
    )
    data_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    data_parser.add_argument(
        "--augmentation-ratio",
        type=float,
        default=0.3,
        help="Proportion of examples to augment",
    )
    data_parser.add_argument(
        "--hard-examples-ratio",
        type=float,
        default=0.05,
        help="Proportion of examples that should be hard examples (max 500)",
    )

    args = parser.parse_args()

    if args.command == "train":
        train(args)
    elif args.command == "evaluate":
        evaluate(args)
    elif args.command == "infer":
        infer(args)
    elif args.command == "generate-data":
        generate_data(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
