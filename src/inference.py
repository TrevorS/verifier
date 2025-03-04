"""
Inference module for using the trained model.
"""

import logging
import re
from typing import List, Optional, Tuple

import config
from src.model import generate_text, load_model
from src.utils import normalize_text

logger = logging.getLogger(__name__)


def prepare_input(text: str, add_instruction_prefix: bool = True) -> str:
    """
    Prepare the input text for inference by cleaning and normalizing.

    Args:
        text (str): Input text - verbal monetary expression
        add_instruction_prefix (bool): Whether to add the instruction prefix

    Returns:
        str: Normalized text ready for inference
    """
    # Normalize the text (lowercase, standardize whitespace)
    normalized_text = normalize_text(text)

    # Remove any special characters that might interfere with model processing
    normalized_text = re.sub(r"[^\w\s.,\-$€£¥]", "", normalized_text)

    # Add instruction prefix to match training format if requested
    if add_instruction_prefix:
        normalized_text = f"{config.INSTRUCTION_PREFIX}: {normalized_text}"

    return normalized_text


def prepare_batch_inputs(texts: List[str], add_instruction_prefix: bool = True) -> List[str]:
    """
    Prepare a batch of input texts for inference.

    Args:
        texts (List[str]): List of input texts to process
        add_instruction_prefix (bool): Whether to add the instruction prefix

    Returns:
        List[str]: List of normalized texts ready for inference
    """
    return [prepare_input(text, add_instruction_prefix) for text in texts]


def run_model_inference(model, tokenizer, input_text: str, use_greedy_decoding: bool = True) -> str:
    """
    Run inference with the model using appropriate decoding strategy.

    Args:
        model: The loaded model
        tokenizer: The model's tokenizer
        input_text (str): Preprocessed input text
        use_greedy_decoding (bool): Whether to use greedy decoding (True) or beam search (False)

    Returns:
        str: Raw model output
    """
    try:
        # Call the generate_text function which returns a dictionary with dollars and cents
        prediction = generate_text(
            model=model,
            tokenizer=tokenizer,
            input_text=input_text,
            max_length=None,  # Use default max length
        )
        
        # Format the output as a string for compatibility with existing code
        dollars = int(prediction["dollars"])
        cents = int(prediction["cents"])
        
        # Format the output as a string (e.g., "amount=1025" for $10.25)
        formatted_output = f"amount={dollars * 100 + cents}"
        
        return formatted_output

    except Exception as e:
        logger.error(f"Error during model inference: {str(e)}")
        return ""


def extract_amount(output_text: str) -> Optional[float]:
    """
    Parse the numeric amount from the model output.

    Args:
        output_text (str): Raw model output (pipe-delimited numeric amount)

    Returns:
        float: Parsed amount, or None if invalid
    """
    # dollars|cents
    # Extract the amount from the output text
    amount_match = re.search(r"amount=(\d+)", output_text)
    if amount_match:
        amount_str = amount_match.group(1)
        try:
            return float(amount_str)
        except ValueError:
            return None


def inference_pipeline(text: str, model_path: str) -> Tuple[Optional[float], str]:
    """
    Complete inference pipeline for processing a verbal monetary expression.

    Args:
        text (str): Input verbal monetary expression
        model_path (str): Path to the trained model

    Returns:
        tuple: (parsed_amount, raw_output)
    """
    # Load model and tokenizer
    model, tokenizer, metadata = load_model(model_path)

    # Prepare input with instruction prefix
    processed_input = prepare_input(text, add_instruction_prefix=True)

    # Run inference
    raw_output = run_model_inference(model, tokenizer, processed_input)

    # Log the raw output
    logger.info(f"Raw model output for input '{text}': {raw_output}")

    # Post-process output
    amount = extract_amount(raw_output)

    # Log result
    if amount:
        logger.info(f"Successfully processed: '{text}' -> {amount}")
    else:
        logger.warning(f"Failed to generate valid amount for: '{text}' -> {raw_output}")

    return amount, raw_output


def run_inference(model_path, texts):
    """
    Run inference with the model on multiple inputs.

    Args:
        model_path (str): Path to the trained model
        texts (list or str): Input text(s) to process

    Returns:
        list: List of amounts
    """
    # Ensure texts is a list
    if isinstance(texts, str):
        texts = [texts]

    # Load the model and tokenizer
    model, tokenizer, metadata = load_model(model_path)

    # Process each input text
    results = []
    for text in texts:
        # Prepare input with instruction prefix
        input_text = prepare_input(text, add_instruction_prefix=True)

        # Generate prediction
        prediction = run_model_inference(model, tokenizer, input_text)

        # Log the raw output
        logger.info(f"Raw model output for input '{text}': {prediction}")

        # Post-process output
        amount = extract_amount(prediction)

        # Add to results
        results.append(amount)

    return results


def demo_inference(model_path: str):
    """
    Demonstrate the inference pipeline with example inputs.

    Args:
        model_path (str): Path to the trained model
    """
    examples = [
        "twenty three dollars and forty five cents",
        "one hundred dollars",
        "five thousand dollars",
        "seven dollars and fifty cents",
        "twenty five dollars and thirty five cents",
        "one thousand two hundred thirty four dollars and fifty six cents",
        "ten dollars",
        "fifty cents",
        "one hundred and fifty dollars",
        "twelve dollars and twenty five cents",
    ]

    print("\n===== INFERENCE DEMO =====")
    print(f"Using model from: {model_path}")
    print("Running inference on example inputs (bank check style verbal amounts in USD)...\n")

    for example in examples:
        print(f"Input: {example}")
        # Using the full inference pipeline which will add the instruction prefix
        amount, raw_output = inference_pipeline(example, model_path)

        print(f"Raw model output: {raw_output}")
        print(f"Amount value: {amount}")
        print("-" * 40)

    print("\nDemo completed.")


if __name__ == "__main__":
    import argparse
    import sys

    # Set up argument parser for CLI
    parser = argparse.ArgumentParser(description="Run inference on verbal monetary expressions")
    parser.add_argument("--model_path", required=True, help="Path to the trained model")
    parser.add_argument("--text", help="Input text to process")
    parser.add_argument("--file", help="File with input texts (one per line)")
    parser.add_argument("--demo", action="store_true", help="Run demo with example inputs")

    args = parser.parse_args()

    if args.demo:
        demo_inference(args.model_path)
    elif args.file:
        with open(args.file, "r") as f:
            texts = [line.strip() for line in f if line.strip()]
        results = run_inference(args.model_path, texts)
        for text, result in zip(texts, results):
            print(f"Input: {text}")
            print(f"Output: {result}")
            print()
    elif args.text:
        amount, raw_output = inference_pipeline(args.text, args.model_path)
        print(f"Input: {args.text}")
        print(f"Output: {raw_output}")
        print(f"Amount value: {amount}")
    else:
        print("Error: Must provide --text, --file, or --demo")
        sys.exit(1)
