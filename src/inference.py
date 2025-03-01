"""
Inference module for using the trained model.
"""

import json
import logging
import re
from typing import Dict, List, Optional, Tuple

from src.model import generate_text, load_model
from src.utils import normalize_text

logger = logging.getLogger(__name__)


def prepare_input(text: str) -> str:
    """
    Prepare the input text for inference by cleaning and normalizing.

    Args:
        text (str): Input text - verbal monetary expression

    Returns:
        str: Normalized text ready for inference
    """
    # Normalize the text (lowercase, standardize whitespace)
    normalized_text = normalize_text(text)

    # Remove any special characters that might interfere with model processing
    normalized_text = re.sub(r"[^\w\s.,\-$€£¥]", "", normalized_text)

    return normalized_text


def prepare_batch_inputs(texts: List[str]) -> List[str]:
    """
    Prepare a batch of input texts for inference.

    Args:
        texts (List[str]): List of input texts to process

    Returns:
        List[str]: List of normalized texts ready for inference
    """
    return [prepare_input(text) for text in texts]


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
        # Set generation parameters based on decoding strategy
        if use_greedy_decoding:
            # Greedy decoding (num_beams=1)
            output_text = generate_text(
                model=model,
                tokenizer=tokenizer,
                input_text=input_text,
                num_beams=1,
                temperature=1.0,  # No randomness in greedy decoding
            )
        else:
            # Beam search for better quality
            output_text = generate_text(model=model, tokenizer=tokenizer, input_text=input_text, num_beams=4, temperature=1.0)

        return output_text

    except Exception as e:
        logger.error(f"Error during model inference: {str(e)}")
        return ""


def extract_json_from_output(output_text: str) -> str:
    """
    Extract JSON string from model output, handling potential text wrapping.

    Args:
        output_text (str): Raw model output

    Returns:
        str: Extracted JSON string
    """
    # Remove any leading/trailing whitespace
    output_text = output_text.strip()

    # Try to extract JSON using regex (finding text between { and })
    json_pattern = r"(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})"
    json_matches = re.findall(json_pattern, output_text)

    if json_matches:
        return json_matches[0]  # Return the first JSON-like string found

    return output_text  # Return as is if no JSON pattern found


def validate_json(json_str: str) -> Tuple[bool, Optional[Dict]]:
    """
    Validate that the string is proper JSON and has expected fields.

    Args:
        json_str (str): JSON string to validate

    Returns:
        tuple: (is_valid, parsed_json_object)
    """
    try:
        # Try to parse as JSON
        json_obj = json.loads(json_str)

        # Check for required fields
        required_fields = ["amount", "currency"]
        if all(field in json_obj for field in required_fields):
            return True, json_obj
        else:
            missing = [field for field in required_fields if field not in json_obj]
            logger.warning(f"JSON missing required fields: {missing}")
            return False, json_obj

    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse as JSON: {json_str}. Error: {str(e)}")
        return False, None


def parse_amount(json_obj: Optional[Dict]) -> Optional[float]:
    """
    Parse the numeric amount from a JSON object.

    Args:
        json_obj (dict): Parsed JSON object

    Returns:
        float: Parsed amount, or None if invalid
    """
    if json_obj is None:
        return None

    try:
        if "amount" in json_obj:
            amount_value = json_obj["amount"]
            # Convert string to float if needed
            if isinstance(amount_value, str):
                # Remove any currency symbols or commas
                clean_amount = re.sub(r"[^\d.-]", "", amount_value)
                return float(clean_amount)
            elif isinstance(amount_value, (int, float)):
                return float(amount_value)
        return None
    except (ValueError, TypeError) as e:
        logger.warning(f"Failed to parse amount: {str(e)}")
        return None


def post_process_output(output_text: str) -> Tuple[str, Optional[Dict], bool, Optional[float]]:
    """
    Post-process the model output to extract and validate JSON.

    Args:
        output_text (str): Raw model output

    Returns:
        tuple: (json_string, parsed_json_object, is_valid, parsed_amount)
    """
    # Extract JSON from output
    json_str = extract_json_from_output(output_text)

    # Validate JSON structure
    is_valid, json_obj = validate_json(json_str)

    # Parse amount if JSON is valid
    amount = parse_amount(json_obj) if json_obj else None

    return json_str, json_obj, is_valid, amount


def inference_pipeline(text: str, model_path: str) -> Tuple[str, Optional[Dict], bool, Optional[float], str]:
    """
    Complete inference pipeline for processing a verbal monetary expression.

    Args:
        text (str): Input verbal monetary expression
        model_path (str): Path to the trained model

    Returns:
        tuple: (json_string, parsed_json_object, is_valid, parsed_amount, raw_output)
    """
    # Load model and tokenizer
    model, tokenizer, metadata = load_model(model_path)

    # Prepare input
    processed_input = prepare_input(text)

    # Run inference
    raw_output = run_model_inference(model, tokenizer, processed_input)

    # Log the raw output
    logger.info(f"Raw model output for input '{text}': {raw_output}")

    # Post-process output
    json_str, json_obj, is_valid, amount = post_process_output(raw_output)

    # Log result
    if is_valid:
        logger.info(f"Successfully processed: '{text}' -> {json_str}")
    else:
        logger.warning(f"Failed to generate valid JSON for: '{text}' -> {raw_output}")

    return json_str, json_obj, is_valid, amount, raw_output


def run_inference(model_path, texts):
    """
    Run inference with the model on multiple inputs.

    Args:
        model_path (str): Path to the trained model
        texts (list or str): Input text(s) to process

    Returns:
        list: List of JSON results
    """
    # Ensure texts is a list
    if isinstance(texts, str):
        texts = [texts]

    # Load the model and tokenizer
    model, tokenizer, metadata = load_model(model_path)

    # Process each input text
    results = []
    for text in texts:
        # Prepare input
        input_text = prepare_input(text)

        # Generate prediction
        prediction = run_model_inference(model, tokenizer, input_text)

        # Log the raw output
        logger.info(f"Raw model output for input '{text}': {prediction}")

        # Post-process output
        json_str, json_obj, is_valid, amount = post_process_output(prediction)

        # Add to results
        results.append(json_str)

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
        json_str, json_obj, is_valid, amount, raw_output = inference_pipeline(example, model_path)

        print(f"Raw model output: {raw_output}")
        print(f"Output JSON: {json_str}")
        if is_valid:
            print(f"Parsed object: {json_obj}")
            print(f"Amount value: {amount}")
            if json_obj.get("currency"):
                print(f"Currency: {json_obj['currency']}")
        else:
            print("Invalid JSON output")
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
        json_str, json_obj, is_valid, amount, raw_output = inference_pipeline(args.text, args.model_path)
        print(f"Input: {args.text}")
        print(f"Output JSON: {json_str}")
        if is_valid:
            print(f"Parsed object: {json_obj}")
            print(f"Amount value: {amount}")
            if json_obj and json_obj.get("currency"):
                print(f"Currency: {json_obj['currency']}")
        else:
            print("Invalid JSON output")
    else:
        print("Error: Must provide --text, --file, or --demo")
        sys.exit(1)
