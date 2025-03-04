#!/usr/bin/env python
"""
Inference script for the character-level transformer model.
"""

import argparse
import logging
import os
import sys

import torch

from src import config
from src.char_model import (
    CharacterEncoder,
    CharacterTransformerModel,
    generate_prediction,
    batch_process
)

# Configure logger
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run inference with a character-level transformer model")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to the saved model")
    parser.add_argument("--device", type=str, default=config.DEVICE,
                        help="Device to run inference on (cuda, mps, or cpu)")
    parser.add_argument("--max_length", type=int, default=200,
                        help="Maximum sequence length")
    parser.add_argument("--force_cpu", action="store_true",
                        help="Force using CPU even if GPU is available")
    
    # Input arguments
    parser.add_argument("--input", type=str, default=None,
                        help="Input text to process")
    parser.add_argument("--input_file", type=str, default=None,
                        help="Path to a file with input texts (one per line)")
    parser.add_argument("--interactive", action="store_true",
                        help="Run in interactive mode")
    
    # Output arguments
    parser.add_argument("--output_file", type=str, default=None,
                        help="Path to save the output predictions")
    
    return parser.parse_args()


def load_model(model_path, device):
    """Load a saved character-level transformer model."""
    logger.info(f"Loading model from {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get model configuration
    model_config = checkpoint["model_config"]
    
    # Initialize model with saved configuration
    model = CharacterTransformerModel(
        vocab_size=model_config["vocab_size"],
        embedding_dim=model_config["embedding_dim"],
        hidden_dim=model_config["hidden_dim"],
        num_heads=model_config["num_heads"],
        num_layers=model_config["num_layers"],
        max_length=model_config["max_length"]
    )
    
    # Load model state
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # Move model to device
    model = model.to(device)
    model.eval()
    
    logger.info(f"Model loaded successfully. Device: {next(model.parameters()).device}")
    
    return model


def process_input(model, char_encoder, input_text, device):
    """Process a single input text."""
    prediction = generate_prediction(model, char_encoder, input_text, device=device)
    
    # Format the output
    dollars = int(prediction["dollars"])
    cents = int(prediction["cents"])
    
    return {
        "input": input_text,
        "prediction": prediction,
        "formatted": f"${dollars}.{cents:02d}"
    }


def process_file(model, char_encoder, input_file, output_file, device):
    """Process a file with input texts."""
    # Read input texts
    with open(input_file, "r") as f:
        input_texts = [line.strip() for line in f if line.strip()]
    
    logger.info(f"Processing {len(input_texts)} inputs from {input_file}")
    
    # Process inputs in batches
    predictions = batch_process(model, char_encoder, input_texts, device=device)
    
    # Format results
    results = []
    for text, pred in zip(input_texts, predictions):
        dollars = int(pred["dollars"])
        cents = int(pred["cents"])
        results.append({
            "input": text,
            "prediction": pred,
            "formatted": f"${dollars}.{cents:02d}"
        })
    
    # Write results to output file if specified
    if output_file:
        import json
        with open(output_file, "w") as f:
            for result in results:
                f.write(json.dumps(result) + "\n")
        logger.info(f"Results saved to {output_file}")
    
    # Print results
    for result in results:
        print(f"Input: {result['input']}")
        print(f"Prediction: {result['formatted']}")
        print()
    
    return results


def interactive_mode(model, char_encoder, device):
    """Run in interactive mode."""
    print("Character-level Transformer Model for Monetary Amount Prediction")
    print("Enter 'q', 'quit', or 'exit' to quit")
    print()
    
    while True:
        # Get input
        try:
            input_text = input("Enter text: ")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break
        
        # Check if user wants to quit
        if input_text.lower() in ["q", "quit", "exit"]:
            print("Exiting...")
            break
        
        # Skip empty input
        if not input_text.strip():
            continue
        
        # Process input
        result = process_input(model, char_encoder, input_text, device)
        
        # Print result
        print(f"Prediction: {result['formatted']}")
        print()


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    # Check if force_cpu is set or if MPS compatibility issues might occur
    if args.force_cpu:
        logger.info("Forcing CPU usage as requested")
        args.device = "cpu"
    elif args.device == "mps" and not os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"):
        logger.warning(
            "MPS device detected but PYTORCH_ENABLE_MPS_FALLBACK environment variable not set. "
            "Some operations may not be supported on MPS. "
            "Consider setting PYTORCH_ENABLE_MPS_FALLBACK=1 or using --force_cpu."
        )
        logger.info("Setting device to CPU to avoid compatibility issues.")
        args.device = "cpu"
    
    # Check if model path is provided
    if args.model_path is None:
        # Try to find the best model in the default location
        default_model_path = os.path.join(config.MODELS_DIR, "char_model", "best_model.pt")
        if os.path.exists(default_model_path):
            args.model_path = default_model_path
        else:
            logger.error("No model path provided and no default model found")
            sys.exit(1)
    
    # Load model
    model = load_model(args.model_path, args.device)
    
    # Initialize character encoder
    char_encoder = CharacterEncoder(max_length=args.max_length)
    
    # Process input based on the provided arguments
    if args.interactive:
        interactive_mode(model, char_encoder, args.device)
    elif args.input_file:
        process_file(model, char_encoder, args.input_file, args.output_file, args.device)
    elif args.input:
        result = process_input(model, char_encoder, args.input, args.device)
        print(f"Input: {result['input']}")
        print(f"Prediction: {result['formatted']}")
    else:
        logger.error("No input provided. Use --input, --input_file, or --interactive")
        sys.exit(1)


if __name__ == "__main__":
    main() 
