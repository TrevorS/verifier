"""
Verifier module for checking the validity of generated records.
"""

import json
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Dict, List, Tuple


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


def float_to_decimal(amount: float) -> Decimal:
    """
    Convert a float to Decimal with proper rounding.

    Args:
        amount (float): The float amount to convert

    Returns:
        Decimal: The converted amount with proper rounding
    """
    # Convert float to string with high precision to avoid float representation issues
    return Decimal(f"{amount:.10f}").quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def verify_record(record: Dict) -> Tuple[bool, str]:
    """
    Verify a single record for floating point rounding errors.

    Args:
        record (Dict): The record to verify containing 'amount' and 'target' fields

    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    try:
        # Get the float amount
        float_amount = record["amount"]

        # Get the delimited target amount
        delimited_amount = record["target"]

        # Convert both to Decimal for precise comparison
        decimal_from_float = float_to_decimal(float_amount)
        decimal_from_delimited = parse_delimited_amount(delimited_amount)

        # Compare the values
        if decimal_from_float != decimal_from_delimited:
            return False, (
                f"Mismatch found: float amount {float_amount} ({decimal_from_float}) "
                f"!= delimited amount {delimited_amount} ({decimal_from_delimited})"
            )

        return True, ""

    except Exception as e:
        return False, f"Error processing record: {str(e)}"


def verify_dataset(file_path: str) -> Tuple[bool, List[str]]:
    """
    Verify all records in a dataset file.

    Args:
        file_path (str): Path to the dataset file

    Returns:
        Tuple[bool, List[str]]: (all_valid, list_of_error_messages)
    """
    file_path = Path(file_path)
    if not file_path.exists():
        return False, [f"File not found: {file_path}"]

    errors = []
    record_count = 0
    error_count = 0

    try:
        with open(file_path, "r") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    record = json.loads(line)
                    record_count += 1

                    is_valid, error_msg = verify_record(record)
                    if not is_valid:
                        error_count += 1
                        errors.append(f"Line {line_num}: {error_msg}")

                except json.JSONDecodeError as e:
                    error_count += 1
                    errors.append(f"Line {line_num}: Invalid JSON - {str(e)}")

    except Exception as e:
        return False, [f"Error reading file: {str(e)}"]

    # Calculate error rate (avoid division by zero)
    error_rate = (error_count / record_count * 100) if record_count > 0 else 0

    # Add summary to errors list
    summary = (
        f"\nVerification Summary:\n"
        f"Total Records: {record_count}\n"
        f"Valid Records: {record_count - error_count}\n"
        f"Invalid Records: {error_count}\n"
        f"Error Rate: {error_rate:.2f}%"
    )
    errors.append(summary)

    return error_count == 0, errors


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Verify generated records for floating point rounding errors")
    parser.add_argument("file_path", help="Path to the dataset file to verify")
    args = parser.parse_args()

    is_valid, errors = verify_dataset(args.file_path)

    # Print all errors
    for error in errors:
        print(error)

    # Exit with appropriate status code
    exit(0 if is_valid else 1)
