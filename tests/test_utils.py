import json

import pytest

from src.utils import format_json, normalize_text, number_to_words


class TestNormalizeText:
    def test_lowercase_conversion(self):
        assert normalize_text("HELLO WORLD") == "hello world"

    def test_whitespace_normalization(self):
        assert normalize_text("  hello   world  ") == "hello world"
        assert normalize_text("hello\t\nworld") == "hello world"

    def test_combined_normalization(self):
        assert normalize_text("  HELLO   WORLD  ") == "hello world"


class TestFormatJSON:
    def test_integer_amount(self):
        result = format_json(5)
        parsed = json.loads(result)
        assert parsed["amount"] == 5.0

    def test_decimal_amount(self):
        result = format_json(25.1)
        parsed = json.loads(result)
        assert parsed["amount"] == 25.1

    def test_two_decimal_precision(self):
        result = format_json(100.015)  # Should round to 100.02
        parsed = json.loads(result)
        assert parsed["amount"] == 100.02

    def test_zero_amount(self):
        result = format_json(0)
        parsed = json.loads(result)
        assert parsed["amount"] == 0.0


class TestNumberToWords:
    @pytest.mark.parametrize(
        "amount,expected",
        [
            (25.10, "twenty-five dollars and ten cents"),
            (5.00, "five dollars"),
            (0.75, "seventy-five cents"),
            (100.01, "one hundred dollars and one cent"),
            (0.01, "one cent"),
            (1.00, "one dollar"),
            (1000.99, "one thousand dollars and ninety-nine cents"),
            (0.00, "zero dollars"),
        ],
    )
    def test_with_and(self, amount, expected):
        assert number_to_words(amount, include_and=True) == expected

    @pytest.mark.parametrize(
        "amount,expected",
        [
            (25.10, "twenty-five dollars ten cents"),
            (5.00, "five dollars"),
            (0.75, "seventy-five cents"),
            (100.01, "one hundred dollars one cent"),
            (0.01, "one cent"),
            (1.00, "one dollar"),
            (1000.99, "one thousand dollars ninety-nine cents"),
            (0.00, "zero dollars"),
        ],
    )
    def test_without_and(self, amount, expected):
        assert number_to_words(amount, include_and=False) == expected

    def test_rounding(self):
        # Test that cents are properly rounded
        assert "twenty-five cents" in number_to_words(0.249)
        assert "twenty-five cents" in number_to_words(0.251)

    def test_hyphenation(self):
        # Test that compound numbers are properly hyphenated
        assert "twenty-two dollars" in number_to_words(22.0)
        assert "one hundred and one dollars" in number_to_words(101.0)
