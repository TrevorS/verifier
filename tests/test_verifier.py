"""Tests for the verifier module."""

import json
from decimal import Decimal

import pytest

from src.verifier import (
    float_to_decimal,
    parse_delimited_amount,
    verify_dataset,
    verify_record,
)


def test_parse_delimited_amount():
    """Test parsing delimited amount strings."""
    # Test basic cases
    assert parse_delimited_amount("123|45") == Decimal("123.45")
    assert parse_delimited_amount("0|00") == Decimal("0.00")
    assert parse_delimited_amount("1000|00") == Decimal("1000.00")

    # Test with different delimiter
    assert parse_delimited_amount("123:45", delimiter=":") == Decimal("123.45")

    # Test edge cases
    assert parse_delimited_amount("0|01") == Decimal("0.01")
    assert parse_delimited_amount("999999|99") == Decimal("999999.99")

    # Test error cases
    with pytest.raises(ValueError):
        parse_delimited_amount("invalid")
    with pytest.raises(ValueError):
        parse_delimited_amount("123.45")  # Wrong format
    with pytest.raises(ValueError):
        parse_delimited_amount("abc|de")  # Non-numeric


def test_float_to_decimal():
    """Test converting floats to Decimal with proper rounding."""
    # Test basic cases
    assert float_to_decimal(123.45) == Decimal("123.45")
    assert float_to_decimal(0.00) == Decimal("0.00")
    assert float_to_decimal(1000.00) == Decimal("1000.00")

    # Test rounding
    assert float_to_decimal(123.456) == Decimal("123.46")  # Round up
    assert float_to_decimal(123.454) == Decimal("123.45")  # Round down
    assert float_to_decimal(123.455) == Decimal("123.46")  # Round half up

    # Test floating point precision issues
    assert float_to_decimal(0.1 + 0.2) == Decimal("0.30")  # Should handle 0.30000000000000004
    assert float_to_decimal(1.1 + 2.2) == Decimal("3.30")  # Should handle 3.3000000000000003


def test_verify_record():
    """Test record verification."""
    # Test valid records
    valid_record = {"amount": 123.45, "target": "123|45"}
    is_valid, error = verify_record(valid_record)
    assert is_valid
    assert error == ""

    # Test invalid records
    invalid_record = {
        "amount": 123.45,
        "target": "123|46",  # Mismatch
    }
    is_valid, error = verify_record(invalid_record)
    assert not is_valid
    assert "Mismatch found" in error

    # Test floating point precision edge cases
    edge_case = {
        "amount": 0.1 + 0.2,  # 0.30000000000000004 in float
        "target": "0|30",
    }
    is_valid, error = verify_record(edge_case)
    assert is_valid
    assert error == ""

    # Test missing fields
    missing_amount = {"target": "123|45"}
    is_valid, error = verify_record(missing_amount)
    assert not is_valid
    assert "Error processing record" in error

    missing_target = {"amount": 123.45}
    is_valid, error = verify_record(missing_target)
    assert not is_valid
    assert "Error processing record" in error


def test_verify_dataset(tmp_path):
    """Test dataset verification."""
    # Create test files
    valid_file = tmp_path / "valid.jsonl"
    invalid_file = tmp_path / "invalid.jsonl"
    mixed_file = tmp_path / "mixed.jsonl"

    # Valid records
    valid_records = [{"amount": 123.45, "target": "123|45"}, {"amount": 0.30, "target": "0|30"}, {"amount": 1000.00, "target": "1000|00"}]

    # Invalid records
    invalid_records = [
        {"amount": 123.45, "target": "123|46"},  # Mismatch
        {"amount": 0.30, "target": "invalid"},  # Invalid format
        {"target": "1000|00"},  # Missing amount
    ]

    # Write test files
    with open(valid_file, "w") as f:
        for record in valid_records:
            f.write(json.dumps(record) + "\n")

    with open(invalid_file, "w") as f:
        for record in invalid_records:
            f.write(json.dumps(record) + "\n")

    with open(mixed_file, "w") as f:
        f.write(json.dumps(valid_records[0]) + "\n")
        f.write(json.dumps(invalid_records[0]) + "\n")
        f.write(json.dumps(valid_records[1]) + "\n")

    # Test valid file
    is_valid, errors = verify_dataset(str(valid_file))
    assert is_valid
    assert len([e for e in errors if not e.startswith("[Summary]")]) == 0

    # Test invalid file
    is_valid, errors = verify_dataset(str(invalid_file))
    assert not is_valid
    assert len([e for e in errors if not e.startswith("[Summary]")]) == 3

    # Test mixed file
    is_valid, errors = verify_dataset(str(mixed_file))
    assert not is_valid
    assert len([e for e in errors if not e.startswith("[Summary]")]) == 1

    # Test non-existent file
    is_valid, errors = verify_dataset("nonexistent.jsonl")
    assert not is_valid
    assert "File not found" in errors[0]


def test_verify_dataset_malformed_json(tmp_path):
    """Test dataset verification with malformed JSON."""
    malformed_file = tmp_path / "malformed.jsonl"

    # Write file with malformed JSON
    with open(malformed_file, "w") as f:
        f.write('{"amount": 123.45, "target": "123|45"}\n')  # Valid
        f.write('{"amount": 123.45, target": "123|45"}\n')  # Invalid JSON
        f.write('{"amount": 456.78, "target": "456|78"}\n')  # Valid

    is_valid, errors = verify_dataset(str(malformed_file))
    assert not is_valid
    assert any("Invalid JSON" in error for error in errors)
    assert len([e for e in errors if not e.startswith("[Summary]")]) == 1


@pytest.mark.parametrize(
    "amount,target,expected_valid",
    [
        (0.01, "0|01", True),
        (9.99, "9|99", True),
        (10.00, "10|00", True),
        (99.99, "99|99", True),
        (100.00, "100|00", True),
        (999.99, "999|99", True),
        (1000.00, "1000|00", True),
        (9999.99, "9999|99", True),
        (0.1 + 0.2, "0|30", True),  # Floating point precision test
        (1.2 + 2.3, "3|50", True),  # Floating point precision test
        (123.45, "123|46", False),  # Mismatch
        (123.456, "123|46", True),  # Correct rounding
        (123.454, "123|45", True),  # Correct rounding
    ],
)
def test_verify_record_parametrized(amount, target, expected_valid):
    """Parametrized tests for various amount formats and edge cases."""
    record = {"amount": amount, "target": target}
    is_valid, _ = verify_record(record)
    assert is_valid == expected_valid
