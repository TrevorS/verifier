"""
Data generation module for creating synthetic training data.
"""

import json
import os
import random
import re
import string
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import inflect
import numpy as np

from src.utils import normalize_text

# Create a global inflect engine for use across all variation functions
p = inflect.engine()

# Dictionary of variation functions
VARIATIONS = {}


def register_variation(name, weight=1.0):
    """
    Decorator to register a variation function.

    Args:
        name (str): The name of the variation
        weight (float): The weight to assign this variation (higher = more frequent)
    """

    def decorator(func):
        VARIATIONS[name] = {"func": func, "weight": weight}
        return func

    return decorator


# Helper functions for variations


def dollars_to_words(dollars):
    """Convert dollar amount to words."""
    return p.number_to_words(dollars)


def cents_to_words(cents):
    """Convert cents amount to words."""
    return p.number_to_words(cents)


def dollars_to_text(dollars):
    """Format dollars with proper pluralization."""
    dollar_words = dollars_to_words(dollars)
    return f"{dollar_words} {p.plural('dollar', dollars)}"


def cents_to_text(cents):
    """Format cents with proper pluralization."""
    cent_words = cents_to_words(cents)
    return f"{cent_words} {p.plural('cent', cents)}"


def format_large_number(number, use_and=True, use_commas=True):
    """Format large numbers with optional 'and' and commas."""
    words = p.number_to_words(number)

    # Add 'and' after hundreds if requested
    if use_and and number > 100:
        words = re.sub(r"hundred (?=\w)", "hundred and ", words)

    # Add or remove commas based on preference
    if use_commas:
        # Ensure commas in millions, thousands, etc.
        words = re.sub(r"million (?=\w)", "million, ", words)
        words = re.sub(r"thousand (?=\w)", "thousand, ", words)
    else:
        # Remove all commas
        words = words.replace(",", "")

    return words.strip()


def get_cents_as_fraction(cents):
    """Format cents as a fraction (xx/100)."""
    return f"{cents}/100"


def unhyphenate_compound(text):
    """Remove hyphens from compound numbers."""
    return text.replace("-", " ")


def is_round_amount(amount):
    """Check if an amount is a whole number of dollars."""
    return amount == int(amount)


# Variation functions


@register_variation("standard", weight=10.0)
def standard_format(amount):
    """
    Standard dollars and cents format.
    Example: 25.10 → "twenty-five dollars and ten cents"
    """
    dollars = int(amount)
    cents = int(round((amount - dollars) * 100))

    result = []

    # Handle dollars part
    if dollars > 0:
        result.append(dollars_to_text(dollars))

    # Handle cents part
    if cents > 0:
        if dollars > 0:
            result.append("and")
        result.append(cents_to_text(cents))

    # Special case for zero amount
    if dollars == 0 and cents == 0:
        return "zero dollars"

    return " ".join(result)


@register_variation("fractional_cents", weight=8.0)
def fractional_cents_format(amount):
    """
    Dollars with cents expressed as a fraction (xx/100).
    Example: 25.10 → "twenty-five dollars and 10/100"
    """
    dollars = int(amount)
    cents = int(round((amount - dollars) * 100))

    result = []

    # Handle dollars part
    if dollars > 0:
        result.append(dollars_to_text(dollars))

    # Handle cents part
    if cents > 0 or (dollars > 0 and cents == 0):  # Include "00/100" if there are dollars
        if dollars > 0:
            result.append("and")
        result.append(get_cents_as_fraction(cents))

    # Special case for zero amount
    if dollars == 0 and cents == 0:
        return "zero dollars"

    return " ".join(result)


@register_variation("dollars_after_fraction", weight=6.0)
def dollars_after_fraction_format(amount):
    """
    Fractional cents with "dollars" at the end.
    Example: 25.10 → "twenty-five and 10/100 dollars"
    """
    dollars = int(amount)
    cents = int(round((amount - dollars) * 100))

    # Special case for zero amount
    if dollars == 0 and cents == 0:
        return "zero dollars"

    if dollars == 0:
        return f"{get_cents_as_fraction(cents)} dollars"

    dollar_words = dollars_to_words(dollars)

    if cents == 0:
        return f"{dollar_words} dollars"
    else:
        return f"{dollar_words} and {get_cents_as_fraction(cents)} dollars"


@register_variation("only_dollars", weight=5.0)
def only_dollars_format(amount):
    """
    "dollars only" for whole dollar amounts.
    Example: 25.00 → "twenty-five dollars only"
    """
    dollars = int(amount)
    cents = int(round((amount - dollars) * 100))

    # Only use this format if there are no cents
    if cents > 0:
        return standard_format(amount)

    if dollars == 0:
        return "zero dollars"

    return f"{dollars_to_text(dollars)} only"


@register_variation("no_cents_specification", weight=4.0)
def no_cents_specification(amount):
    """
    Omits cents if the value is zero.
    Example: 25.00 → "twenty-five dollars"
    """
    dollars = int(amount)
    cents = int(round((amount - dollars) * 100))

    if dollars == 0 and cents == 0:
        return "zero dollars"

    if cents == 0:
        return dollars_to_text(dollars)

    return standard_format(amount)


@register_variation("no_and", weight=4.0)
def no_and_format(amount):
    """
    Omits the "and" between dollars and cents.
    Example: 25.10 → "twenty-five dollars ten cents"
    """
    dollars = int(amount)
    cents = int(round((amount - dollars) * 100))

    result = []

    # Handle dollars part
    if dollars > 0:
        result.append(dollars_to_text(dollars))

    # Handle cents part
    if cents > 0:
        result.append(cents_to_text(cents))

    # Special case for zero amount
    if dollars == 0 and cents == 0:
        return "zero dollars"

    return " ".join(result)


@register_variation("non_hyphenated", weight=3.0)
def non_hyphenated_format(amount):
    """
    Compound numbers without hyphens.
    Example: 25.10 → "twenty five dollars and ten cents"
    """
    # Get standard format then remove hyphens
    standard = standard_format(amount)
    return unhyphenate_compound(standard)


@register_variation("with_and_in_numbers", weight=3.0)
def with_and_in_numbers_format(amount):
    """
    Includes "and" in numbers over 100.
    Example: 125.10 → "one hundred and twenty-five dollars and ten cents"
    """
    dollars = int(amount)
    cents = int(round((amount - dollars) * 100))

    if dollars == 0 and cents == 0:
        return "zero dollars"

    result = []

    # Handle dollars with "and" in numbers over 100
    if dollars > 0:
        dollar_words = format_large_number(dollars, use_and=True)
        result.append(f"{dollar_words} {p.plural('dollar', dollars)}")

    # Handle cents
    if cents > 0:
        if dollars > 0:
            result.append("and")
        cent_words = cents_to_words(cents)
        result.append(f"{cent_words} {p.plural('cent', cents)}")

    return " ".join(result)


@register_variation("zero_cents_explicit", weight=2.0)
def zero_cents_explicit_format(amount):
    """
    Explicitly states "zero cents".
    Example: 25.00 → "twenty-five dollars and zero cents"
    """
    dollars = int(amount)
    cents = int(round((amount - dollars) * 100))

    # Special case for zero amount
    if dollars == 0 and cents == 0:
        return "zero dollars and zero cents"

    result = []

    # Handle dollars part
    if dollars > 0:
        result.append(dollars_to_text(dollars))

    # Always append cents part, even if zero
    if dollars > 0:
        result.append("and")

    if cents > 0:
        result.append(cents_to_text(cents))
    else:
        result.append("zero cents")

    return " ".join(result)


@register_variation("even_dollars", weight=2.0)
def even_dollars_format(amount):
    """
    Uses "even" for zero cents.
    Example: 25.00 → "twenty-five dollars even"
    """
    dollars = int(amount)
    cents = int(round((amount - dollars) * 100))

    # Only use if there are dollars and no cents
    if cents > 0 or dollars == 0:
        return standard_format(amount)

    return f"{dollars_to_text(dollars)} even"


@register_variation("comma_instead_of_and", weight=2.0)
def comma_instead_of_and_format(amount):
    """
    Uses a comma instead of "and".
    Example: 25.10 → "twenty-five dollars, ten cents"
    """
    dollars = int(amount)
    cents = int(round((amount - dollars) * 100))

    result = []

    # Handle dollars part
    if dollars > 0:
        result.append(dollars_to_text(dollars))

    # Handle cents part
    if cents > 0:
        if dollars > 0:
            result.append(",")
        result.append(cents_to_text(cents))

    # Special case for zero amount
    if dollars == 0 and cents == 0:
        return "zero dollars"

    return " ".join(result)


@register_variation("ampersand", weight=1.5)
def ampersand_format(amount):
    """
    Uses an ampersand (&) instead of "and".
    Example: 25.10 → "twenty-five dollars & ten cents"
    """
    dollars = int(amount)
    cents = int(round((amount - dollars) * 100))

    result = []

    # Handle dollars part
    if dollars > 0:
        result.append(dollars_to_text(dollars))

    # Handle cents part
    if cents > 0:
        if dollars > 0:
            result.append("&")
        result.append(cents_to_text(cents))

    # Special case for zero amount
    if dollars == 0 and cents == 0:
        return "zero dollars"

    return " ".join(result)


@register_variation("with_preposition", weight=1.5)
def with_preposition_format(amount):
    """
    Uses "with" instead of "and".
    Example: 25.10 → "twenty-five dollars with ten cents"
    """
    dollars = int(amount)
    cents = int(round((amount - dollars) * 100))

    result = []

    # Handle dollars part
    if dollars > 0:
        result.append(dollars_to_text(dollars))

    # Handle cents part
    if cents > 0:
        if dollars > 0:
            result.append("with")
        result.append(cents_to_text(cents))

    # Special case for zero amount
    if dollars == 0 and cents == 0:
        return "zero dollars"

    return " ".join(result)


@register_variation("formal_business", weight=2.0)
def formal_business_format(amount):
    """
    A very formal phrasing, starting with "the sum of".
    Example: 25478.90 → "the sum of twenty-five thousand four hundred seventy-eight dollars and ninety cents"
    """
    standard = standard_format(amount)
    return f"the sum of {standard}"


@register_variation("cents_only", weight=2.0)
def cents_only_format(amount):
    """
    Express only in cents: "X cents" (only for amounts < 1.00)
    Example: 0.75 → "seventy-five cents"
    """
    dollars = int(amount)
    cents = int(round((amount - dollars) * 100))

    # Only use this format for amounts less than $1.00
    if dollars > 0:
        return standard_format(amount)

    if cents == 0:
        return "zero cents"

    return cents_to_text(cents)


@register_variation("and_no_cents", weight=1.5)
def and_no_cents_format(amount):
    """
    Uses "and no cents".
    Example: 25.00 → "twenty-five dollars and no cents"
    """
    dollars = int(amount)
    cents = int(round((amount - dollars) * 100))

    # Only use if there are dollars and no cents
    if cents > 0 or dollars == 0:
        return standard_format(amount)

    return f"{dollars_to_text(dollars)} and no cents"


@register_variation("legal_document", weight=1.0)
def legal_document_format(amount):
    """
    Includes the amount in parentheses after the words.
    Example: 1250.75 → "one thousand two hundred fifty dollars and seventy-five cents ($1,250.75)"
    """
    standard = standard_format(amount)
    formatted_amount = "${:,.2f}".format(amount)
    return f"{standard} ({formatted_amount})"


@register_variation("exactly", weight=2.0)
def exact_amount_format(amount):
    """
    Includes "exactly" at the beginning.
    Example: 8547.32 → "exactly eight thousand five hundred forty-seven dollars and thirty-two cents"
    """
    standard = standard_format(amount)
    return f"exactly {standard}"


@register_variation("hundreds_as_multiples", weight=1.0)
def hundreds_as_multiples_format(amount):
    """
    Expresses hundreds as multiples (e.g., "fifteen hundred").
    Example: 1500.00 → "fifteen hundred dollars"
    """
    dollars = int(amount)
    cents = int(round((amount - dollars) * 100))

    # Only applicable for amounts between 1100 and 9900 (11 to 99 hundred)
    if not (1100 <= dollars < 10000) or (dollars % 100 != 0):
        return standard_format(amount)

    # Express as X hundred
    hundreds = dollars // 100
    hundred_words = p.number_to_words(hundreds)

    if cents == 0:
        return f"{hundred_words} hundred dollars"
    else:
        cent_words = cents_to_words(cents)
        return f"{hundred_words} hundred dollars and {cent_words} {p.plural('cent', cents)}"


@register_variation("extra_commas", weight=1.0)
def extra_commas_format(amount):
    """
    Inserts extra commas within the number words.
    Example: 8765.43 → "eight thousand, seven hundred, sixty-five dollars, and forty-three cents"
    """
    dollars = int(amount)
    cents = int(round((amount - dollars) * 100))

    if dollars == 0 and cents == 0:
        return "zero dollars"

    result = []

    # Handle dollars with extra commas
    if dollars > 0:
        dollar_words = format_large_number(dollars, use_and=False, use_commas=True)
        # Add extra commas
        dollar_words = re.sub(r"(\w+) (\w+)", r"\1, \2", dollar_words)
        result.append(f"{dollar_words} {p.plural('dollar', dollars)}")

    # Handle cents
    if cents > 0:
        if dollars > 0:
            result.append(",")
            result.append("and")
        cent_words = cents_to_words(cents)
        result.append(f"{cent_words} {p.plural('cent', cents)}")

    return " ".join(result)


@register_variation("written_and_numeric", weight=1.0)
def written_and_numeric_format(amount):
    """
    Repeats the amount parenthetically in numeric format.
    Example: 5432.10 → "five thousand four hundred thirty-two and 10/100 dollars ($5,432.10)"
    """
    # Get fractional cents format
    fractional = dollars_after_fraction_format(amount)
    formatted_amount = "${:,.2f}".format(amount)
    return f"{fractional} ({formatted_amount})"


@register_variation("payment_of", weight=1.0)
def payment_of_format(amount):
    """
    Begins with "payment of".
    Example: 3456.78 → "payment of three thousand four hundred fifty-six and 78/100 dollars"
    """
    fractional = dollars_after_fraction_format(amount)
    return f"payment of {fractional}"


@register_variation("pay_to", weight=1.0)
def pay_to_format(amount):
    """
    Uses the phrase "pay to the order of".
    Example: 4321.98 → "pay to the order of four thousand three hundred twenty-one and 98/100 dollars"
    """
    fractional = dollars_after_fraction_format(amount)
    return f"pay to the order of {fractional}"


@register_variation("remittance", weight=0.8)
def remittance_format(amount):
    """
    Begins with "remittance of".
    Example: 5432.10 → "remittance of five thousand four hundred thirty-two and 10/100 dollars"
    """
    fractional = dollars_after_fraction_format(amount)
    return f"remittance of {fractional}"


@register_variation("funds_transfer", weight=0.8)
def funds_transfer_format(amount):
    """
    Specifies a "transfer of funds".
    Example: 9876.54 → "transfer of funds in the amount of nine thousand eight hundred seventy-six and 54/100 dollars"
    """
    fractional = dollars_after_fraction_format(amount)
    return f"transfer of funds in the amount of {fractional}"


@register_variation("dollars_and_xx_100", weight=5.0)
def dollars_and_xx_100_format(amount):
    """
    Explicit "xx/100" for cents, even if zero.
    Example: 25.00 → "twenty-five dollars and 00/100"
    """
    dollars = int(amount)
    cents = int(round((amount - dollars) * 100))

    if dollars == 0 and cents == 0:
        return "zero dollars and 00/100"

    result = []

    # Handle dollars part
    if dollars > 0:
        result.append(dollars_to_text(dollars))

    # Always include fractional cents
    if dollars > 0:
        result.append("and")

    result.append(f"{cents:02d}/100")

    return " ".join(result)


@register_variation("no_commas", weight=2.0)
def no_commas_format(amount):
    """
    Omits commas in the written number.
    Example: 1234567.89 → "one million two hundred thirty-four thousand five hundred sixty-seven dollars and eighty-nine cents"
    """
    dollars = int(amount)
    cents = int(round((amount - dollars) * 100))

    if dollars == 0 and cents == 0:
        return "zero dollars"

    result = []

    # Handle dollars with no commas
    if dollars > 0:
        dollar_words = format_large_number(dollars, use_and=False, use_commas=False)
        result.append(f"{dollar_words} {p.plural('dollar', dollars)}")

    # Handle cents
    if cents > 0:
        if dollars > 0:
            result.append("and")
        cent_words = cents_to_words(cents)
        result.append(f"{cent_words} {p.plural('cent', cents)}")

    return " ".join(result)


@register_variation("parenthetical_cents", weight=0.8)
def parenthetical_cents_format(amount):
    """
    Puts the cents in parentheses.
    Example: 9876.54 → "nine thousand eight hundred seventy-six dollars (and fifty-four cents)"
    """
    dollars = int(amount)
    cents = int(round((amount - dollars) * 100))

    if dollars == 0 and cents == 0:
        return "zero dollars"

    if dollars == 0:
        return cents_to_text(cents)

    result = f"{dollars_to_text(dollars)}"

    if cents > 0:
        cent_text = cents_to_text(cents)
        result += f" (and {cent_text})"

    return result


@register_variation("amount_in_words", weight=0.8)
def amount_in_words_format(amount):
    """
    Uses the prefix "amount in words:".
    Example: 3456.78 → "amount in words: three thousand four hundred fifty-six and 78/100 dollars"
    """
    fractional = dollars_after_fraction_format(amount)
    return f"amount in words: {fractional}"


@register_variation("asterisk_bounded", weight=0.5)
def asterisk_bounded_format(amount):
    """
    Surrounds the amount with asterisks.
    Example: 5432.10 → "*five thousand four hundred thirty-two and 10/100 dollars*"
    """
    fractional = dollars_after_fraction_format(amount)
    return f"*{fractional}*"


@register_variation("no_dollars", weight=0.8)
def no_dollars_format(amount):
    """
    Uses "no dollars" instead of "zero dollars".
    Example: 0.50 → "no dollars and fifty cents"
    """
    dollars = int(amount)
    cents = int(round((amount - dollars) * 100))

    if dollars > 0:
        return standard_format(amount)

    if cents == 0:
        return "no dollars"

    return f"no dollars and {cents_to_text(cents)}"


@register_variation("high_value_transaction", weight=0.5)
def high_value_transaction_format(amount):
    """
    Precise format for very large amounts.
    Example: 25000000.00 → "twenty-five million dollars and 00/100"
    """
    dollars = int(amount)
    cents = int(round((amount - dollars) * 100))

    # Only for truly large amounts
    if dollars < 1000000:
        return standard_format(amount)

    dollar_words = format_large_number(dollars, use_and=True, use_commas=True)

    if cents == 0:
        return f"{dollar_words} dollars and 00/100"
    else:
        return f"{dollar_words} dollars and {cents:02d}/100"


@register_variation("equals", weight=0.3)
def equals_format(amount):
    """
    Uses "equals" to connect dollars and cents.
    Example: 5678.90 → "five thousand six hundred seventy-eight dollars equals ninety cents"
    """
    dollars = int(amount)
    cents = int(round((amount - dollars) * 100))

    if dollars == 0 and cents == 0:
        return "zero dollars"

    if dollars == 0:
        return cents_to_text(cents)

    if cents == 0:
        return dollars_to_text(dollars)

    return f"{dollars_to_text(dollars)} equals {cents_to_text(cents)}"


@register_variation("personal_check_standard", weight=3.0)
def personal_check_standard_format(amount):
    """
    Standard personal check format.
    Example: 123.45 → "one hundred twenty-three and 45/100 dollars"
    """
    dollars = int(amount)
    cents = int(round((amount - dollars) * 100))

    if dollars == 0 and cents == 0:
        return "zero and 00/100 dollars"

    if dollars == 0:
        return f"{cents:02d}/100 dollars"

    dollar_words = dollars_to_words(dollars)
    return f"{dollar_words} and {cents:02d}/100 dollars"


@register_variation("bill_payment", weight=2.0)
def bill_payment_format(amount):
    """
    Bill payment format (personal to business).
    Example: 89.95 → "eighty-nine dollars and ninety-five cents only"
    """
    standard = standard_format(amount)
    return f"{standard} only"


@register_variation("donation_check", weight=1.5)
def donation_check_format(amount):
    """
    Donation check format.
    Example: 50.00 → "fifty dollars and 00/100"
    """
    dollars = int(amount)
    cents = int(round((amount - dollars) * 100))

    if dollars == 0 and cents == 0:
        return "zero dollars and 00/100"

    result = []

    if dollars > 0:
        result.append(dollars_to_text(dollars))

    result.append("and")
    result.append(f"{cents:02d}/100")

    return " ".join(result)


# Add more variations here...


def generate_random_amount(min_amount=0.01, max_amount=1000000.00):
    """
    Generate a random monetary amount within the specified range.

    Args:
        min_amount (float): Minimum amount (inclusive, must be positive)
        max_amount (float): Maximum amount (inclusive, must be positive)

    Returns:
        float: A random positive monetary amount

    Raises:
        ValueError: If min_amount or max_amount is not positive
    """
    # Validate that parameters are positive
    if min_amount <= 0:
        raise ValueError("min_amount must be positive (> 0)")

    if max_amount <= 0:
        raise ValueError("max_amount must be positive (> 0)")

    # Generate amount using log-uniform distribution to ensure good coverage
    # across orders of magnitude (e.g., cents, dollars, thousands, etc.)
    log_min = np.log10(min_amount)
    log_max = np.log10(max_amount)

    # Generate a random value in log space
    log_amount = random.uniform(log_min, log_max)

    # Convert back to linear space
    amount = 10**log_amount

    # Round to 2 decimal places (cents)
    amount = round(amount * 100) / 100

    return amount


def generate_stratified_amounts(num_examples):
    """
    Generate a stratified sample of monetary amounts across different ranges.

    Args:
        num_examples (int): Number of examples to generate

    Returns:
        list: List of monetary amounts with good distribution
    """
    amounts = []

    # Define ranges for stratification
    ranges = [
        (0.01, 0.99),  # Cents only
        (1.00, 9.99),  # Single-digit dollars
        (10.00, 99.99),  # Double-digit dollars
        (100.00, 999.99),  # Triple-digit dollars
        (1000.00, 9999.99),  # Thousands
        (10000.00, 99999.99),  # Tens of thousands
        (100000.00, 1000000.00),  # Hundreds of thousands to million
    ]

    # Assign more weight to common ranges (cents, single, double, triple-digit dollars)
    weights = [0.15, 0.20, 0.20, 0.15, 0.15, 0.10, 0.05]

    # Determine number of examples per range
    counts = [int(num_examples * w) for w in weights]

    # Adjust to ensure we get exactly num_examples
    remainder = num_examples - sum(counts)
    counts[0] += remainder

    # Generate amounts for each range
    for (min_val, max_val), count in zip(ranges, counts):
        for _ in range(count):
            amounts.append(generate_random_amount(min_val, max_val))

    # Shuffle the amounts
    random.shuffle(amounts)

    return amounts


def amount_to_verbal_expression(amount, variation_type=None):
    """
    Convert a numerical monetary amount to a verbal expression with variations.

    Args:
        amount (float): Monetary amount (must be non-negative)
        variation_type (str, optional): Type of variation to generate
            If None, a random variation will be chosen based on weights.

    Returns:
        tuple: (str, str) - (verbal expression of the monetary amount, variation name used)

    Raises:
        ValueError: If amount is negative or variation_type is not recognized
    """
    # Validate that amount is non-negative
    if amount < 0:
        raise ValueError("amount must be non-negative (>= 0)")

    # If variation type is specified, use it
    if variation_type is not None:
        if variation_type in VARIATIONS:
            return normalize_text(VARIATIONS[variation_type]["func"](amount)), variation_type
        else:
            raise ValueError(f"Unknown variation type: {variation_type}")

    # Otherwise, choose a random variation based on weights
    weights = [VARIATIONS[var_name]["weight"] for var_name in VARIATIONS]
    variation_name = random.choices(list(VARIATIONS.keys()), weights=weights, k=1)[0]

    return normalize_text(VARIATIONS[variation_name]["func"](amount)), variation_name


def apply_augmentation(text, dropout_prob=0.05, case_change_prob=0.2):
    """
    Apply text augmentation to the input text.

    Args:
        text (str): Input text
        dropout_prob (float): Probability of dropping a character
        case_change_prob (float): Probability of changing case (before normalization)

    Returns:
        str: Augmented text
    """
    # Apply case change before dropout (will be normalized later)
    if random.random() < case_change_prob:
        if random.random() < 0.5:
            # Uppercase a random word
            words = text.split()
            if words:
                idx = random.randint(0, len(words) - 1)
                words[idx] = words[idx].upper()
                text = " ".join(words)
        else:
            # Random capitalization
            text = "".join(c.upper() if random.random() < 0.3 else c for c in text)

    # Character dropout
    if random.random() < 0.3:  # Only apply dropout to 30% of samples
        chars = []
        for c in text:
            if c in string.whitespace or random.random() > dropout_prob:
                chars.append(c)
        text = "".join(chars)

    # Add typo (character substitution)
    if random.random() < 0.1:  # Apply to 10% of samples
        chars = list(text)
        if chars:
            idx = random.randint(0, len(chars) - 1)
            if chars[idx] not in string.whitespace:
                # Replace with a random character
                chars[idx] = random.choice(string.ascii_lowercase)
            text = "".join(chars)

    # Add extra whitespace
    if random.random() < 0.2:  # Apply to 20% of samples
        words = text.split()
        if words:
            idx = random.randint(0, len(words) - 1)
            words[idx] = "  " + words[idx] + "  "
            text = " ".join(words)

    # Normalize the text (lowercase and standardize whitespace)
    return normalize_text(text)


def generate_examples(num_examples, control_variation_distribution=True):
    """
    Generate examples of verbal monetary expressions and their JSON representations.

    Args:
        num_examples (int): Number of examples to generate
        control_variation_distribution (bool): Whether to manually control variation distribution

    Returns:
        list: List of dictionaries with input and target fields
    """
    # Generate stratified amounts
    amounts = generate_stratified_amounts(num_examples)

    examples = []

    if control_variation_distribution:
        # Define target distribution (adjust weights as needed)
        variation_targets = {
            "standard": 0.25,  # Most common
            "fractional_cents": 0.15,  # Common in business
            "dollars_after_fraction": 0.10,  # Common in formal writing
            "no_and": 0.08,  # Common spoken form
            "only_dollars": 0.05,  # For whole dollars
            "cents_only": 0.05,  # For amounts < $1
            # Other variations get smaller percentages
        }

        # Fill in remaining variations with equal small weights
        remaining_weight = 1.0 - sum(variation_targets.values())
        remaining_variations = [v for v in VARIATIONS.keys() if v not in variation_targets]
        for var in remaining_variations:
            variation_targets[var] = remaining_weight / len(remaining_variations)

        # Determine counts
        variation_counts = {var: int(num_examples * weight) for var, weight in variation_targets.items()}

        # Adjust for rounding errors
        total_assigned = sum(variation_counts.values())
        if total_assigned < num_examples:
            variation_counts["standard"] += num_examples - total_assigned

        # Create variation assignment list
        variations_list = []
        for var, count in variation_counts.items():
            variations_list.extend([var] * count)

        # Shuffle variations
        random.shuffle(variations_list)

        # Match amounts with variations
        for amount, variation in zip(amounts, variations_list):
            # Skip incompatible combinations
            if variation == "cents_only" and amount >= 1:
                variation = "standard"  # Fallback
            if variation == "only_dollars" and amount % 1 != 0:
                variation = "standard"  # Fallback

            # Generate verbal expression
            verbal_expr, _ = amount_to_verbal_expression(amount, variation_type=variation)

            # Create example
            examples.append(
                {"input": verbal_expr, "target": create_target_output(amount), "amount": float(format(amount, ".2f")), "variation": variation}
            )
    else:
        # Original approach - random selection based on weights
        for amount in amounts:
            # Generate verbal expression (with random variation)
            verbal_expr, variation_name = amount_to_verbal_expression(amount)

            # Create example
            examples.append(
                {
                    "input": verbal_expr,
                    "target": create_target_output(amount),
                    "amount": float(format(amount, ".2f")),
                    "variation": variation_name,
                }
            )

    return examples


def create_target_output(amount, delimiter="|"):
    """
    Create a target output for a given amount using precise decimal arithmetic.
    This function handles cases like 9.995 correctly (rounds to 10.00) and avoids
    common floating-point issues (e.g. 1.6099999999999999 becoming 1.60 instead of 1.61).

    Args:
        amount (float): The numeric value to format.
        delimiter (str): The delimiter to use between dollars and cents.

    Returns:
        str: A string representing the amount in 'dollars|cents' format.
    """
    # Convert the float to a string with high precision then to Decimal.
    # This helps to mitigate float precision errors.
    d_amount = Decimal(f"{amount:.10f}")

    # Round to two decimal places using ROUND_HALF_UP
    rounded_amount = d_amount.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    # Split into dollars and cents
    dollars, cents = str(rounded_amount).split(".")
    return delimiter.join([dollars, cents])


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
    examples = generate_examples(num_examples)

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

    print(f"Generated {len(train_examples)} training examples and {len(val_examples)} validation examples")
    print(f"Training data saved to: {train_path}")
    print(f"Validation data saved to: {val_path}")

    # Analyze variation distribution
    variation_count = {name: 0 for name in VARIATIONS.keys()}

    for example in examples:
        variation = example.get("variation", "unknown")
        if variation in variation_count:
            variation_count[variation] += 1
        else:
            variation_count["unknown"] = variation_count.get("unknown", 0) + 1

    # Print variation distribution
    print("\nVariation distribution:")
    for var_name, count in sorted(variation_count.items(), key=lambda x: x[1], reverse=True):
        if count > 0:
            percentage = (count / num_examples) * 100
            print(f"{var_name}: {count} examples ({percentage:.1f}%)")

    # Sample examples for inspection
    sample_size = min(10, num_examples)
    sample_examples = random.sample(examples, sample_size)

    print("\nSample examples:")
    for i, example in enumerate(sample_examples, 1):
        print(f"\nExample {i}:")
        print(f"Input:  {example['input']}")
        print(f"Target: {example['target']}")
        print(f"Amount: ${example['amount']:.2f}")
        print(f"Variation: {example['variation']}")

    return train_path, val_path


def create_complete_dataset(
    num_examples=100000,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    output_dir=None,
    seed=42,
    augmentation_ratio=0.3,
    hard_examples_ratio=0.05,  # 5% of examples should be hard examples
):
    """Create a complete dataset with train, validation, and test splits."""
    import math

    import datasets

    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)

    # Verify split ratios sum to 1
    if not math.isclose(train_ratio + val_ratio + test_ratio, 1.0, abs_tol=1e-10):
        raise ValueError("Split ratios must sum to 1")

    # Set up output directory
    if output_dir is None:
        output_dir = Path(__file__).parents[1] / "data"
    else:
        output_dir = Path(output_dir)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Calculate split sizes
    train_size = int(num_examples * train_ratio)
    val_size = int(num_examples * val_ratio)
    test_size = num_examples - train_size - val_size

    # Calculate how many hard examples to generate
    num_hard_examples = min(int(num_examples * hard_examples_ratio), 500)

    print(f"Generating {num_examples} examples:")
    print(f"- Training: {train_size}")
    print(f"- Validation: {val_size}")
    print(f"- Test: {test_size}")
    print(f"- Including {num_hard_examples} hard examples ({hard_examples_ratio * 100:.1f}%)")

    # Generate hard examples first
    if num_hard_examples > 0:
        print(f"Generating {num_hard_examples} hard examples...")
        hard_examples = generate_hard_examples(num_hard_examples)
    else:
        hard_examples = []

    # Calculate how many regular examples we need
    num_regular_examples = num_examples - len(hard_examples)

    # Generate regular examples
    print(f"Generating {num_regular_examples} regular examples...")
    regular_examples = generate_examples(num_regular_examples, control_variation_distribution=True)

    # Add additional metadata
    print("Adding metadata...")
    for example in regular_examples:
        # Add amount range category
        amount = example["amount"]
        example["amount_range"] = classify_amount_range(amount)

        # Add complexity indicator
        example["complexity"] = classify_complexity(example["input"])

        # Add has_cents flag
        example["has_cents"] = (amount % 1) != 0

        # Apply augmentation to some examples
        if random.random() < augmentation_ratio:
            example["input"] = apply_augmentation(example["input"])
            example["augmented"] = True
        else:
            example["augmented"] = False

    # Combine all examples
    all_examples = regular_examples + hard_examples

    # Verify we have the correct total number of examples
    assert len(all_examples) == num_examples, f"Expected {num_examples} total examples, got {len(all_examples)}"

    # Shuffle all examples
    random.shuffle(all_examples)

    # Split into train, validation, and test sets
    train_examples = all_examples[:train_size]
    val_examples = all_examples[train_size : train_size + val_size]
    test_examples = all_examples[train_size + val_size : train_size + val_size + test_size]

    # Verify we have the correct number of examples in each split
    assert len(train_examples) == train_size, f"Expected {train_size} training examples, got {len(train_examples)}"
    assert len(val_examples) == val_size, f"Expected {val_size} validation examples, got {len(val_examples)}"
    assert len(test_examples) == test_size, f"Expected {test_size} test examples, got {len(test_examples)}"

    # Save to files
    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"
    test_path = output_dir / "test.jsonl"

    # Write data
    print("Writing files...")
    with open(train_path, "w") as f:
        for example in train_examples:
            f.write(json.dumps(example) + "\n")

    with open(val_path, "w") as f:
        for example in val_examples:
            f.write(json.dumps(example) + "\n")

    with open(test_path, "w") as f:
        for example in test_examples:
            f.write(json.dumps(example) + "\n")

    print(f"Files saved to {output_dir}")

    # Create HuggingFace dataset
    print("Creating HuggingFace dataset...")
    dataset = datasets.load_dataset("json", data_files={"train": str(train_path), "validation": str(val_path), "test": str(test_path)})

    # Print dataset statistics
    print_dataset_statistics(dataset)

    # Display sample examples
    display_sample_examples(dataset, num_examples=5)

    return dataset, (train_path, val_path, test_path)


def print_dataset_statistics(dataset):
    """Print detailed statistics about the dataset."""
    print("\nDataset Statistics:")

    # Overall counts
    for split in dataset:
        print(f"- {split.capitalize()}: {len(dataset[split])} examples")

    # Variation distribution
    print("\nVariation Distribution (Top 10):")
    for split in dataset:
        variation_counts = {}
        for example in dataset[split]:
            variation = example["variation"]
            variation_counts[variation] = variation_counts.get(variation, 0) + 1

        print(f"\n{split.capitalize()} split:")
        for variation, count in sorted(variation_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            percentage = (count / len(dataset[split])) * 100
            print(f"  {variation}: {count} examples ({percentage:.1f}%)")

    # Amount range distribution
    print("\nAmount Range Distribution:")
    for split in dataset:
        range_counts = {}
        for example in dataset[split]:
            amount_range = example["amount_range"]
            range_counts[amount_range] = range_counts.get(amount_range, 0) + 1

        print(f"\n{split.capitalize()} split:")
        for range_name, count in sorted(range_counts.items()):
            percentage = (count / len(dataset[split])) * 100
            print(f"  {range_name}: {count} examples ({percentage:.1f}%)")


def display_sample_examples(dataset, num_examples=5):
    """Display a sample of examples from each split."""
    print("\nSample Examples:")

    for split in dataset:
        print(f"\n{split.capitalize()} Split Examples:")

        # Get random indices
        indices = random.sample(range(len(dataset[split])), min(num_examples, len(dataset[split])))

        for i, idx in enumerate(indices):
            example = dataset[split][idx]

            print(f"\nExample {i + 1}:")
            print(f"Input: {example['input']}")
            print(f"Target: {example['target']}")
            print(f"Amount: ${example['amount']:.2f}")
            print(f"Variation: {example['variation']}")
            print(f"Amount Range: {example['amount_range']}")
            print(f"Complexity: {example['complexity']}")
            print(f"Has Cents: {'Yes' if example['has_cents'] else 'No'}")
            print(f"Augmented: {'Yes' if example.get('augmented', False) else 'No'}")


def generate_hard_examples(num_hard_examples=500):
    """Generate a set of challenging examples to test specific edge cases."""
    hard_examples = []

    # Define challenging amounts
    challenging_amounts = [
        0.01,  # One cent
        0.25,  # Quarter
        0.99,  # Just under a dollar
        1.00,  # Exactly one dollar
        1.01,  # Just over a dollar
        9.99,  # Common price point
        100.00,  # Even hundred
        1000.00,  # Even thousand
        1234.56,  # Sequential digits
        10000.00,  # Even ten thousand
        1000000.00,  # Million
        1000000.01,  # Just over a million
    ]

    # Important variations to test with every challenging amount
    key_variations = ["standard", "fractional_cents", "dollars_after_fraction", "no_and", "with_and_in_numbers", "legal_document"]

    # Generate combinations
    for amount in challenging_amounts:
        for variation in key_variations:
            # Skip incompatible combinations
            if variation == "cents_only" and amount >= 1:
                continue

            # Generate verbal expression
            verbal_expr, _ = amount_to_verbal_expression(amount, variation_type=variation)

            # Create example
            hard_examples.append(
                {
                    "input": verbal_expr,
                    "target": create_target_output(amount),
                    "amount": float(format(amount, ".2f")),
                    "variation": variation,
                    "hard_example": True,  # Mark as hard example
                    "amount_range": classify_amount_range(amount),
                    "has_cents": (amount % 1) != 0,
                    "complexity": classify_complexity(verbal_expr),
                    "augmented": False,
                }
            )

    # Cap at num_hard_examples
    random.shuffle(hard_examples)
    return hard_examples[:num_hard_examples]


def classify_amount_range(amount):
    """Helper to classify an amount into a range category."""
    if amount < 1:
        return "cents_only"
    elif amount < 10:
        return "single_digit"
    elif amount < 100:
        return "double_digit"
    elif amount < 1000:
        return "triple_digit"
    elif amount < 10000:
        return "thousands"
    elif amount < 100000:
        return "tens_thousands"
    else:
        return "hundreds_thousands_plus"


def classify_complexity(text):
    """Helper to classify text complexity."""
    if "thousand" in text or "million" in text:
        return "complex"
    elif "hundred" in text:
        return "medium"
    else:
        return "simple"


if __name__ == "__main__":
    # Generate a small sample dataset for testing
    num_examples = 1000
    train_path, val_path = generate_dataset(num_examples=num_examples)

    # Load and analyze the distribution of amounts
    amount_ranges = {
        "cents_only": 0,
        "single_digit": 0,
        "double_digit": 0,
        "triple_digit": 0,
        "thousands": 0,
        "tens_thousands": 0,
        "hundreds_thousands+": 0,
    }

    all_examples = []
    # Read both files
    for path in [train_path, val_path]:
        with open(path, "r") as f:
            for line in f:
                example = json.loads(line)
                all_examples.append(example)

    # Count examples in each range
    for example in all_examples:
        amount = example["amount"]
        if amount < 1:
            amount_ranges["cents_only"] += 1
        elif amount < 10:
            amount_ranges["single_digit"] += 1
        elif amount < 100:
            amount_ranges["double_digit"] += 1
        elif amount < 1000:
            amount_ranges["triple_digit"] += 1
        elif amount < 10000:
            amount_ranges["thousands"] += 1
        elif amount < 100000:
            amount_ranges["tens_thousands"] += 1
        else:
            amount_ranges["hundreds_thousands+"] += 1

    # Print distribution summary
    print("\nAmount range distribution:")
    for range_name, count in amount_ranges.items():
        percentage = (count / num_examples) * 100
        print(f"{range_name}: {count} examples ({percentage:.1f}%)")

    # Print a summary of all available variations
    print("\nAvailable variations:")
    for var_name, var_data in VARIATIONS.items():
        print(f"- {var_name} (weight: {var_data['weight']})")
