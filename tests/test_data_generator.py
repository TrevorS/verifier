import json
import tempfile

import pytest

from src.data_generator import (
    amount_to_verbal_expression,
    apply_augmentation,
    create_target_output,
    generate_dataset,
    generate_examples,
    generate_random_amount,
    generate_stratified_amounts,
)


class TestGenerateRandomAmount:
    def test_output_within_range(self):
        min_amount = 0.01
        max_amount = 1000.00
        for _ in range(100):
            amount = generate_random_amount(min_amount, max_amount)
            assert min_amount <= amount <= max_amount

    def test_output_precision(self):
        # Test that amounts have at most 2 decimal places
        for _ in range(100):
            amount = generate_random_amount()
            # Check if it has at most 2 decimal places
            assert abs(amount - round(amount, 2)) < 1e-10

    def test_negative_min_amount_handling(self):
        # Test that negative min_amount raises ValueError
        with pytest.raises(ValueError, match="min_amount must be positive"):
            generate_random_amount(-10.0, 100.0)

        # Test that negative max_amount raises ValueError
        with pytest.raises(ValueError, match="max_amount must be positive"):
            generate_random_amount(10.0, -100.0)


class TestGenerateStratifiedAmounts:
    def test_output_count(self):
        num_examples = 100
        amounts = generate_stratified_amounts(num_examples)
        assert len(amounts) == num_examples

    def test_distribution(self):
        num_examples = 1000
        amounts = generate_stratified_amounts(num_examples)

        # Check if we have examples in each range
        cents_only = sum(1 for a in amounts if a < 1)
        single_digit = sum(1 for a in amounts if 1 <= a < 10)
        double_digit = sum(1 for a in amounts if 10 <= a < 100)
        triple_digit = sum(1 for a in amounts if 100 <= a < 1000)
        thousands = sum(1 for a in amounts if 1000 <= a < 10000)
        tens_thousands = sum(1 for a in amounts if 10000 <= a < 100000)
        hundreds_thousands = sum(1 for a in amounts if a >= 100000)

        # Assert that each range has at least some examples
        assert cents_only > 0
        assert single_digit > 0
        assert double_digit > 0
        assert triple_digit > 0
        assert thousands > 0
        assert tens_thousands > 0
        assert hundreds_thousands > 0


class TestAmountToVerbalExpression:
    @pytest.mark.parametrize(
        "amount,variation,expected_substr",
        [
            (25.10, "standard", "twenty-five dollars and ten cents"),
            (5.00, "only_dollars", "five dollars only"),
            (0.75, "cents_only", "seventy-five cents"),
            (100.01, "no_and", "one hundred dollars one cent"),
            (42.00, "zero_cents_explicit", "forty-two dollars and zero cents"),
            (1234.56, "standard", "one thousand, two hundred and thirty-four dollars and fifty-six cents"),
            (10000.00, "no_cents_specification", "ten thousand dollars"),
            (1.25, "standard", "one dollar and twenty-five cents"),
            (100.50, "standard", "one hundred dollars and fifty cents"),
        ],
    )
    def test_variation_types(self, amount, variation, expected_substr):
        # The function now returns a tuple of (expression, variation_name)
        result, variation_name = amount_to_verbal_expression(amount, variation_type=variation)
        assert variation_name == variation
        assert expected_substr in result

    def test_negative_amount_handling(self):
        # Negative amounts should raise ValueError
        with pytest.raises(ValueError):
            amount_to_verbal_expression(-25.10)

    def test_random_variation(self):
        # Test that random variation selection works with a positive USD amount
        amount = 25.10
        # Call multiple times to cover different random variations
        variations = set()
        for _ in range(100):
            result, variation_name = amount_to_verbal_expression(amount)
            variations.add(variation_name)
        # We should get at least 2 different variations
        assert len(variations) >= 2


class TestApplyAugmentation:
    def test_augmentation_preserves_meaning(self):
        # Instead of testing a single output and hoping it doesn't have
        # specific augmentations, we'll do a statistical test to make sure
        # that most outputs preserve key words.
        text = "twenty-five dollars and ten cents"
        preserved_count = 0
        num_tests = 20

        for _ in range(num_tests):
            augmented = apply_augmentation(text)
            # We'll check if any of these key words are preserved
            key_word_preserved = any(word in augmented for word in ["dollar", "cent", "twenty", "five", "ten"])
            if key_word_preserved:
                preserved_count += 1

        # Make sure at least 80% of the tests preserve some meaning
        assert preserved_count >= 0.8 * num_tests

    def test_normalization(self):
        text = "TWENTY-FIVE dollars AND ten CENTS"
        augmented = apply_augmentation(text)
        # Result should be lowercase
        assert augmented.islower()
        # Should not have consecutive spaces
        assert "  " not in augmented


class TestGenerateExamples:
    def test_example_format(self):
        num_examples = 10
        examples = generate_examples(num_examples)
        assert len(examples) == num_examples

        for example in examples:
            assert "input" in example
            assert "target" in example
            assert "amount" in example
            assert "variation" in example  # Check for variation field

            # Check target format
            assert "dollars" in example["target"]
            assert "cents" in example["target"]
            assert isinstance(example["target"]["dollars"], float)
            assert isinstance(example["target"]["cents"], float)

            # Check that amount field matches target values
            expected_amount = example["target"]["dollars"] + (example["target"]["cents"] / 100)
            assert abs(example["amount"] - expected_amount) < 1e-10

            # Check that variation is a non-empty string
            assert isinstance(example["variation"], str)
            assert example["variation"] != ""


class TestGenerateDataset:
    def test_dataset_generation(self):
        # Use a temporary directory for test output
        with tempfile.TemporaryDirectory() as temp_dir:
            num_examples = 20
            train_ratio = 0.8

            # Generate dataset
            train_path, val_path = generate_dataset(num_examples=num_examples, output_dir=temp_dir, train_ratio=train_ratio)

            # Check paths
            assert train_path.exists()
            assert val_path.exists()

            # Check file contents
            train_examples = []
            val_examples = []

            with open(train_path) as f:
                for line in f:
                    train_examples.append(json.loads(line))

            with open(val_path) as f:
                for line in f:
                    val_examples.append(json.loads(line))

            # Check example counts
            assert len(train_examples) == int(num_examples * train_ratio)
            assert len(val_examples) == num_examples - int(num_examples * train_ratio)

            # Check example format
            for example in train_examples + val_examples:
                assert "input" in example
                assert "target" in example
                assert "amount" in example
                
                # Check target format
                assert "dollars" in example["target"]
                assert "cents" in example["target"]
                assert isinstance(example["target"]["dollars"], float)
                assert isinstance(example["target"]["cents"], float)

                # Check that amount field matches target values
                expected_amount = example["target"]["dollars"] + (example["target"]["cents"] / 100)
                assert abs(example["amount"] - expected_amount) < 1e-10


class TestCreateTargetOutput:
    @pytest.mark.parametrize(
        "amount,expected_dollars,expected_cents",
        [
            (3.07, 3.0, 7.0),  # Example 1
            (5929.70, 5929.0, 70.0),  # Example 2
            (7.82, 7.0, 82.0),  # Example 3
            (68397.46, 68397.0, 46.0),  # Example 4
            (8.68, 8.0, 68.0),  # Example 5
            # Edge cases
            (0.01, 0.0, 1.0),
            (0.001, 0.0, 0.0),  # Rounds down to 0 cents
            (0.995, 1.0, 0.0),  # Rounds up to a dollar
            (9.995, 10.0, 0.0),  # Rounds up and changes dollar digit
        ],
    )
    def test_target_output_matches_displayed_amount(self, amount, expected_dollars, expected_cents):
        # Verify that the target matches what we expect for the formatted amount
        target = create_target_output(amount)

        # Test our target is correct
        assert target["dollars"] == expected_dollars
        assert target["cents"] == expected_cents

        # Double-check by manually calculating what should be in target using the same algorithm
        # as in create_target_output
        # Handle special values like 9.995 that should round up to 10.00
        if abs(amount - round(amount)) < 0.011 and abs(amount - round(amount)) > 0.0001:
            # Only apply this special handling for values close to the next dollar
            fractional_part = abs(amount) % 1
            if fractional_part > 0.99:  # Only for values like 0.995, 9.995, etc.
                rounded_amount = round(amount)  # This will round 9.995 to 10.0
            else:
                # Normal case - round to two decimal places
                rounded_amount = round(amount * 100) / 100
        else:
            # Normal case - round to two decimal places
            rounded_amount = round(amount * 100) / 100

        dollars = int(rounded_amount)
        cents = int(round((rounded_amount - dollars) * 100))

        # Handle the case where cents round to 100
        if cents == 100:
            dollars += 1
            cents = 0

        assert target["dollars"] == float(dollars)
        assert target["cents"] == float(cents)
