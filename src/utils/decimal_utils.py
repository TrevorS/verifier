"""
Utility functions for handling decimal arithmetic and float comparisons.
"""

from decimal import ROUND_HALF_UP, Decimal
from typing import Optional, Tuple, Union


def float_to_decimal(amount: float) -> Decimal:
    """
    Convert a float to Decimal with proper rounding.

    Args:
        amount (float): The float amount to convert

    Returns:
        Decimal: The converted amount with proper rounding to 2 decimal places
    """
    # Convert float to string with high precision to avoid float representation issues
    return Decimal(f"{amount:.10f}").quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def parse_delimited_amount(delimited_str: str, delimiter: str = "|") -> Decimal:
    """
    Parse a delimited amount string (e.g. "123|45") into a Decimal.

    Args:
        delimited_str (str): The delimited string to parse
        delimiter (str): The delimiter used (default: "|")

    Returns:
        Decimal: The parsed amount

    Raises:
        ValueError: If the input string is not in the correct format or contains non-numeric values
    """
    try:
        dollars, cents = delimited_str.split(delimiter)

        # Check if both parts are numeric
        if not dollars.isdigit() or not cents.isdigit():
            raise ValueError("Both dollars and cents must be numeric")

        # Check cents is exactly 2 digits
        if len(cents) != 2:
            raise ValueError("Cents must be exactly 2 digits")

        return Decimal(f"{dollars}.{cents}")
    except ValueError as e:
        raise ValueError(f"Invalid amount format: {str(e)}")
    except Exception:
        raise ValueError(f"Invalid amount format: {delimited_str}")


def compare_amounts(
    amount1: Union[float, Decimal, str],
    amount2: Union[float, Decimal, str],
    tolerance: Optional[Decimal] = None,
) -> Tuple[bool, Optional[Decimal]]:
    """
    Compare two amounts with optional tolerance, handling various input formats.

    Args:
        amount1: First amount (float, Decimal, or delimited string)
        amount2: Second amount (float, Decimal, or delimited string)
        tolerance: Optional tolerance for comparison (as Decimal)

    Returns:
        Tuple[bool, Optional[Decimal]]: (is_equal, difference)
        - is_equal: True if amounts are equal within tolerance
        - difference: Absolute difference between amounts, or None if comparison failed

    Raises:
        ValueError: If either amount is of an unsupported type
    """
    # Convert amounts to Decimal
    if isinstance(amount1, float):
        decimal1 = float_to_decimal(amount1)
    elif isinstance(amount1, str):
        decimal1 = parse_delimited_amount(amount1)
    elif isinstance(amount1, Decimal):
        decimal1 = amount1.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    else:
        raise ValueError(f"Unsupported type for amount1: {type(amount1)}")

    if isinstance(amount2, float):
        decimal2 = float_to_decimal(amount2)
    elif isinstance(amount2, str):
        decimal2 = parse_delimited_amount(amount2)
    elif isinstance(amount2, Decimal):
        decimal2 = amount2.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    else:
        raise ValueError(f"Unsupported type for amount2: {type(amount2)}")

    try:
        # Calculate absolute difference
        difference = abs(decimal1 - decimal2)

        # Compare with tolerance if provided
        if tolerance is not None:
            return difference <= tolerance, difference
        else:
            return difference == 0, difference

    except Exception:
        return False, None


def format_amount(amount: Union[float, Decimal], delimiter: str = "|") -> str:
    """
    Format an amount as a delimited string (e.g. "123|45").

    Args:
        amount: The amount to format
        delimiter: The delimiter to use (default: "|")

    Returns:
        str: The formatted amount string

    Raises:
        ValueError: If the amount is of an unsupported type
    """
    if isinstance(amount, float):
        decimal_amount = float_to_decimal(amount)
    elif isinstance(amount, Decimal):
        decimal_amount = amount.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    else:
        raise ValueError(f"Unsupported type for amount: {type(amount)}")

    dollars, cents = str(decimal_amount).split(".")
    return f"{dollars}{delimiter}{cents}"
