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
    text = re.sub(r"\s+", " ", text).strip()

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
    formatted_amount = float(format(amount, ".2f"))

    # Create JSON object
    json_obj = {"amount": formatted_amount}

    # Convert to JSON string
    return json.dumps(json_obj)


def number_to_words(amount, include_and=True):
    """
    Convert a numerical monetary amount to a verbal expression.

    Args:
        amount (float): Monetary amount
        include_and (bool): Whether to include 'and' between dollars and cents

    Returns:
        str: Verbal expression of the monetary amount
    """
    p = inflect.engine()

    # Split amount into dollars (whole number) and cents (decimals)
    dollars = int(amount)
    cents = int(round((amount - dollars) * 100))

    result = []

    # Handle dollars part
    if dollars > 0:
        dollar_words = p.number_to_words(dollars)
        dollar_text = f"{dollar_words} {p.plural('dollar', dollars)}"
        result.append(dollar_text)

    # Handle cents part
    if cents > 0:
        # Add 'and' between dollars and cents if dollars > 0 and include_and is True
        if dollars > 0 and include_and:
            result.append("and")

        cent_words = p.number_to_words(cents)
        cent_text = f"{cent_words} {p.plural('cent', cents)}"
        result.append(cent_text)

    # Special case for zero dollars and zero cents
    if dollars == 0 and cents == 0:
        return "zero dollars"

    # Handle case where amount is only cents (no dollars)
    if dollars == 0 and cents > 0:
        result = [p.number_to_words(cents), p.plural("cent", cents)]

    # Join parts and normalize
    return normalize_text(" ".join(result))
