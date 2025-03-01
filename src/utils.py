"""
Utility functions for the monetary expressions to JSON converter.
"""
import json
import re

import inflect


def normalize_text(text):
    """
    Normalize text by converting to lowercase and standardizing whitespace.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Normalized text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Standardize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def format_json(amount):
    """
    Format a numeric amount as a JSON string.
    
    Args:
        amount (float): Monetary amount
        
    Returns:
        str: JSON string representation
    """
    # Ensure amount has two decimal places
    formatted_amount = float(format(amount, '.2f'))
    
    # Create JSON object
    json_obj = {"amount": formatted_amount}
    
    # Convert to JSON string
    return json.dumps(json_obj) 