"""Tests for decimal utils module."""

from decimal import Decimal

import pytest

from src.utils.decimal_utils import (
    compare_amounts,
    float_to_decimal,
    format_amount,
    parse_delimited_amount,
)


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

    # Test edge cases
    assert float_to_decimal(0.001) == Decimal("0.00")  # Round to zero
    assert float_to_decimal(9.995) == Decimal("10.00")  # Round up to next dollar
    assert float_to_decimal(0.995) == Decimal("1.00")  # Round up to dollar


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
    with pytest.raises(ValueError):
        parse_delimited_amount("123|5")  # Cents not 2 digits
    with pytest.raises(ValueError):
        parse_delimited_amount("123|456")  # Cents too many digits


def test_compare_amounts():
    """Test comparing amounts with various input formats."""
    # Test exact matches with different input types
    assert compare_amounts(123.45, "123|45")[0] is True
    assert compare_amounts(Decimal("123.45"), 123.45)[0] is True
    assert compare_amounts("123|45", "123|45")[0] is True

    # Test with tolerance
    assert compare_amounts(123.45, 123.46, tolerance=Decimal("0.01"))[0] is True
    assert compare_amounts(123.45, 123.47, tolerance=Decimal("0.01"))[0] is False

    # Test floating point precision issues
    assert compare_amounts(0.1 + 0.2, "0|30")[0] is True
    assert compare_amounts(1.1 + 2.2, "3|30")[0] is True

    # Test differences
    is_equal, diff = compare_amounts(123.45, 123.47)
    assert is_equal is False
    assert diff == Decimal("0.02")

    # Test error cases
    with pytest.raises(ValueError, match="Invalid amount format"):
        compare_amounts("invalid", 123.45)
    with pytest.raises(ValueError, match="Invalid amount format"):
        compare_amounts(123.45, "abc|de")
    with pytest.raises(ValueError, match="Unsupported type"):
        compare_amounts([1, 2, 3], 123.45)


def test_format_amount():
    """Test formatting amounts as delimited strings."""
    # Test basic cases
    assert format_amount(123.45) == "123|45"
    assert format_amount(Decimal("123.45")) == "123|45"
    assert format_amount(0.00) == "0|00"
    assert format_amount(1000.00) == "1000|00"

    # Test with different delimiter
    assert format_amount(123.45, delimiter=":") == "123:45"

    # Test rounding
    assert format_amount(123.456) == "123|46"  # Round up
    assert format_amount(123.454) == "123|45"  # Round down
    assert format_amount(123.455) == "123|46"  # Round half up

    # Test floating point precision issues
    assert format_amount(0.1 + 0.2) == "0|30"  # Should handle 0.30000000000000004
    assert format_amount(1.1 + 2.2) == "3|30"  # Should handle 3.3000000000000003

    # Test edge cases
    assert format_amount(0.001) == "0|00"  # Round to zero
    assert format_amount(9.995) == "10|00"  # Round up to next dollar
    assert format_amount(0.995) == "1|00"  # Round up to dollar

    # Test error cases
    with pytest.raises(ValueError):
        format_amount("123.45")  # String input not supported
    with pytest.raises(ValueError):
        format_amount([1, 2, 3])  # Invalid type
