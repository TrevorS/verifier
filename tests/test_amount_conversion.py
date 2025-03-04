import sys
from pathlib import Path

# Add the parent directory to the path to import from src
sys.path.append(str(Path(__file__).parent.parent))

from src.data_generator import create_target_output


def test_standard_value():
    assert create_target_output(1.61) == "1|61"
    assert create_target_output(59.0) == "59|00"


def test_rounding_up_boundary():
    assert create_target_output(9.995) == "10|00"
    assert create_target_output(0.995) == "1|00"
    assert create_target_output(5.995) == "6|00"


def test_floating_point_imprecision():
    # Verify that common floating-point imprecision issues are handled correctly
    assert create_target_output(1.6099999999999999) == "1|61"
    assert create_target_output(2.2199999999999998) == "2|22"


def test_small_values():
    # Test values near zero to verify proper rounding at very small amounts
    assert create_target_output(0.0049) == "0|00"
    assert create_target_output(0.005) == "0|01"
    assert create_target_output(0.0051) == "0|01"


def test_exact_half():
    # Test values that fall exactly at the half-cent boundary
    assert create_target_output(1.004) == "1|00"
    assert create_target_output(1.005) == "1|01"


def test_negative_values():
    # Test that negative values are formatted correctly
    assert create_target_output(-1.61) == "-1|61"
    assert create_target_output(-0.995) == "-1|00"
    assert create_target_output(-9.995) == "-10|00"


def test_different_delimiters():
    # Test that the function handles custom delimiters correctly
    assert create_target_output(42.42, ",") == "42,42"
    assert create_target_output(42.42, ".") == "42.42"
    assert create_target_output(42.42, "-") == "42-42"
    assert create_target_output(42.42, " ") == "42 42"
    assert create_target_output(42.42, ":") == "42:42"


def test_high_precision_input():
    # Test inputs with many decimal places to ensure proper rounding
    assert create_target_output(1.2345678901) == "1|23"
    # The formatting in the function ensures 10 decimal places, then rounds:
    # 9.876543210987654321 becomes "9.8765432110" and then rounds to "9|88"
    assert create_target_output(9.876543210987654321) == "9|88"


def test_large_values():
    # Test very large numbers to ensure they are handled properly
    assert create_target_output(123456789.987) == "123456789|99"
