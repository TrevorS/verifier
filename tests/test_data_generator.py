import json
import tempfile

import pytest

from src.data_generator import (
    amount_to_verbal_expression,
    apply_augmentation,
    create_json_output,
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
        # Test that negative min_amount is handled correctly
        amount = generate_random_amount(-10.0, 100.0)
        assert amount >= 0.01


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
            (5.00, "dollars_only", "five dollars"),
            (0.75, "cents_only", "seventy-five cents"),
            (100.01, "no_and", "one hundred dollars one cent"),
            (42.00, "with_cents", "forty-two dollars and zero cents"),
        ],
    )
    def test_variation_types(self, amount, variation, expected_substr):
        result = amount_to_verbal_expression(amount, variation_type=variation)
        assert expected_substr in result

    def test_random_variation(self):
        # Test that random variation selection works
        amount = 25.10
        # Call multiple times to cover different random variations
        variations = set()
        for _ in range(100):
            result = amount_to_verbal_expression(amount)
            variations.add(result)
        # We should get at least 2 different variations
        assert len(variations) >= 2


class TestCreateJsonOutput:
    def test_json_format(self):
        amount = 42.73
        json_str = create_json_output(amount)
        parsed = json.loads(json_str)
        assert "amount" in parsed
        assert parsed["amount"] == 42.73

    def test_decimal_precision(self):
        # Test that the amount has 2 decimal places
        amount = 100.456
        json_str = create_json_output(amount)
        parsed = json.loads(json_str)
        assert parsed["amount"] == 100.46  # Should be rounded


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

            # Check that target is valid JSON
            target_json = json.loads(example["target"])
            assert "amount" in target_json

            # Check that amount field matches JSON amount
            assert abs(example["amount"] - target_json["amount"]) < 1e-10


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
