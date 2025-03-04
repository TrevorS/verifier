"""
Verifier module for checking the validity of generated records.
"""

import json
from typing import Dict, List, Tuple

from src.utils.decimal_utils import float_to_decimal, parse_delimited_amount


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
    Verify all records in a dataset for floating point rounding errors.

    Args:
        file_path (str): Path to the dataset file

    Returns:
        Tuple[bool, List[str]]: (is_valid, error_messages)
    """
    errors = []
    total_records = 0
    invalid_records = 0
    has_json_errors = False

    try:
        with open(file_path, "r") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    record = json.loads(line)
                    total_records += 1

                    is_valid, error = verify_record(record)
                    if not is_valid:
                        invalid_records += 1
                        errors.append(f"Line {line_num}: {error}")

                except json.JSONDecodeError as e:
                    has_json_errors = True
                    errors.append(f"Line {line_num}: Invalid JSON - {str(e)}")
                except Exception as e:
                    errors.append(f"Line {line_num}: Error processing record - {str(e)}")

    except FileNotFoundError:
        errors.append(f"File not found: {file_path}")
        return False, errors
    except Exception as e:
        errors.append(f"Error reading file: {str(e)}")
        return False, errors

    # Calculate error rate
    error_rate = invalid_records / total_records if total_records > 0 else 1.0

    # Add summary to errors with prefix
    errors.insert(0, f"[Summary] Processed {total_records} records")
    errors.insert(1, f"[Summary] Found {invalid_records} invalid records")
    errors.insert(2, f"[Summary] Error rate: {error_rate:.2%}")

    # Return False if there are any invalid records or JSON errors
    return not (invalid_records > 0 or has_json_errors), errors


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
