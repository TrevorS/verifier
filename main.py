#!/usr/bin/env python3
"""
Main entry point for the monetary expressions to JSON converter.
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
    logger.info(f"Using model: {config.MODEL_NAME}")
    logger.info(f"Training data: {config.TRAIN_DATA_PATH}")
    logger.info(f"Validation data: {config.VAL_DATA_PATH}")
    
    # Import here to avoid loading modules unnecessarily
    from src.trainer import train_model
    
    train_model(
        model_name=config.MODEL_NAME,
        train_data_path=config.TRAIN_DATA_PATH,
        val_data_path=config.VAL_DATA_PATH,
        output_dir=args.output_dir,
        wandb_logging=not args.no_wandb,
    )
    logger.info("Training completed.")


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
    from src.inference import run_inference
    
    if args.input_file:
        logger.info(f"Input file: {args.input_file}")
        with open(args.input_file, 'r') as f:
            texts = [line.strip() for line in f]
    else:
        texts = [args.text]
    
    results = run_inference(
        model_path=args.model_path,
        texts=texts,
    )
    
    for text, result in zip(texts, results):
        print(f"Input: {text}")
        print(f"Output: {result}")
    
    logger.info("Inference completed.")


def generate_data(args):
    """Generate synthetic training data."""
    logger.info("Generating synthetic data...")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Import here to avoid loading modules unnecessarily
    from src.data_generator import generate_dataset
    
    generate_dataset(
        num_examples=args.num_examples,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
    )
    logger.info("Data generation completed.")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="A sequence-to-sequence model that converts verbal monetary expressions to JSON."
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Train parser
    train_parser = subparsers.add_parser("train", help="Train the model")
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
    
    # Data generation parser
    data_parser = subparsers.add_parser("generate-data", help="Generate synthetic data")
    data_parser.add_argument(
        "--num-examples",
        type=int,
        default=10000,
        help="Number of examples to generate",
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
        help="Ratio of examples to use for training (vs validation)",
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
