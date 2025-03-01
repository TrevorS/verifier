"""
Inference module for using the trained model.
"""
import json
import logging

from src.model import load_model, generate_text
from src.utils import normalize_text

logger = logging.getLogger(__name__)


def prepare_input(text):
    """
    Prepare the input text for inference.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Normalized text ready for inference
    """
    # Normalize the text
    normalized_text = normalize_text(text)
    
    return normalized_text


def post_process_output(output_text):
    """
    Post-process the model output.
    
    Args:
        output_text (str): Raw model output
        
    Returns:
        tuple: JSON string and parsed JSON object (or None if invalid)
    """
    # Remove any leading/trailing whitespace
    output_text = output_text.strip()
    
    # Try to parse as JSON
    try:
        json_obj = json.loads(output_text)
        valid_json = True
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse as JSON: {output_text}")
        json_obj = None
        valid_json = False
    
    return output_text, json_obj, valid_json


def run_inference(model_path, texts):
    """
    Run inference with the model.
    
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
    model, tokenizer = load_model(model_path)
    
    # Process each input text
    results = []
    for text in texts:
        # Prepare input
        input_text = prepare_input(text)
        
        # Generate prediction
        prediction = generate_text(model, tokenizer, input_text)
        
        # Post-process output
        output_text, json_obj, valid_json = post_process_output(prediction)
        
        # Log result
        if valid_json:
            logger.info(f"Successfully processed: {text} -> {output_text}")
        else:
            logger.warning(f"Failed to generate valid JSON for: {text} -> {output_text}")
        
        # Add to results
        results.append(output_text)
    
    return results


def parse_amount(json_text):
    """
    Parse the amount from a JSON string.
    
    Args:
        json_text (str): JSON string
        
    Returns:
        float: Parsed amount, or None if invalid
    """
    try:
        json_obj = json.loads(json_text)
        
        if "amount" in json_obj:
            return json_obj["amount"]
        else:
            return None
    except:
        return None 