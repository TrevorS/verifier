"""
Tests for the inference pipeline module.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.inference import (
    extract_json_from_output,
    inference_pipeline,
    parse_amount,
    post_process_output,
    prepare_batch_inputs,
    prepare_input,
    run_model_inference,
    validate_json,
)


class TestInputPreparation:
    """Tests for input preparation functions."""

    def test_prepare_input(self):
        """Test that input text is properly normalized."""
        # Test basic normalization
        assert prepare_input("TWENTY dollars") == "twenty dollars"

        # Test special character removal
        assert prepare_input("$25.50!") == "$25.50"

        # Test whitespace normalization
        assert prepare_input("  five   hundred  dollars  ") == "five hundred dollars"

        # Test currency symbol preservation
        assert prepare_input("€100") == "€100"
        assert prepare_input("$50.25") == "$50.25"
        assert prepare_input("£10.99") == "£10.99"

    def test_prepare_batch_inputs(self):
        """Test that batch inputs are properly processed."""
        inputs = ["$20", "thirty euros", "ONE HUNDRED YEN"]
        expected = ["$20", "thirty euros", "one hundred yen"]

        assert prepare_batch_inputs(inputs) == expected


class TestModelInference:
    """Tests for model inference functions."""

    @patch("src.inference.generate_text")
    def test_run_model_inference_greedy(self, mock_generate_text):
        """Test greedy decoding inference."""
        # Mock the generate_text function
        mock_generate_text.return_value = '{"amount": 20, "currency": "USD"}'

        # Mock model and tokenizer
        model = MagicMock()
        tokenizer = MagicMock()

        # Run inference with greedy decoding
        result = run_model_inference(model, tokenizer, "twenty dollars", use_greedy_decoding=True)

        # Check the result
        assert result == '{"amount": 20, "currency": "USD"}'

        # Check that generate_text was called with the right parameters
        mock_generate_text.assert_called_once()
        args, kwargs = mock_generate_text.call_args
        assert kwargs["num_beams"] == 1  # Greedy decoding
        assert kwargs["temperature"] == 1.0

    @patch("src.inference.generate_text")
    def test_run_model_inference_beam_search(self, mock_generate_text):
        """Test beam search inference."""
        # Mock the generate_text function
        mock_generate_text.return_value = '{"amount": 20, "currency": "USD"}'

        # Mock model and tokenizer
        model = MagicMock()
        tokenizer = MagicMock()

        # Run inference with beam search
        result = run_model_inference(model, tokenizer, "twenty dollars", use_greedy_decoding=False)

        # Check the result
        assert result == '{"amount": 20, "currency": "USD"}'

        # Check that generate_text was called with the right parameters
        mock_generate_text.assert_called_once()
        args, kwargs = mock_generate_text.call_args
        assert kwargs["num_beams"] == 4  # Beam search
        assert kwargs["temperature"] == 1.0

    @patch("src.inference.generate_text")
    def test_run_model_inference_error_handling(self, mock_generate_text):
        """Test error handling during inference."""
        # Mock generate_text to raise an exception
        mock_generate_text.side_effect = Exception("Test error")

        # Mock model and tokenizer
        model = MagicMock()
        tokenizer = MagicMock()

        # Run inference
        result = run_model_inference(model, tokenizer, "twenty dollars")

        # Check that an empty string is returned
        assert result == ""


class TestPostProcessing:
    """Tests for output post-processing functions."""

    def test_extract_json_from_output(self):
        """Test JSON extraction from model output."""
        # Clean JSON
        clean_json = '{"amount": 20, "currency": "USD"}'
        assert extract_json_from_output(clean_json) == clean_json

        # JSON with text before and after
        messy_json = 'Here is the result: {"amount": 20, "currency": "USD"} as requested.'
        assert extract_json_from_output(messy_json) == '{"amount": 20, "currency": "USD"}'

        # Nested JSON
        nested_json = '{"result": {"amount": 20, "currency": "USD"}}'
        assert extract_json_from_output(nested_json) == nested_json

        # No JSON found
        no_json = "No JSON here"
        assert extract_json_from_output(no_json) == no_json

    def test_validate_json(self):
        """Test JSON validation."""
        # Valid JSON with required fields
        valid_json = '{"amount": 20, "currency": "USD"}'
        is_valid, json_obj = validate_json(valid_json)
        assert is_valid is True
        assert json_obj["amount"] == 20
        assert json_obj["currency"] == "USD"

        # Valid JSON missing required fields
        missing_field = '{"amount": 20}'
        is_valid, json_obj = validate_json(missing_field)
        assert is_valid is False
        assert json_obj["amount"] == 20

        # Invalid JSON
        invalid_json = '{amount: 20, "currency": "USD"}'
        is_valid, json_obj = validate_json(invalid_json)
        assert is_valid is False
        assert json_obj is None

    def test_parse_amount(self):
        """Test amount parsing."""
        # Integer amount
        json_obj = {"amount": 20, "currency": "USD"}
        assert parse_amount(json_obj) == 20.0

        # Float amount
        json_obj = {"amount": 20.5, "currency": "USD"}
        assert parse_amount(json_obj) == 20.5

        # String amount
        json_obj = {"amount": "20.5", "currency": "USD"}
        assert parse_amount(json_obj) == 20.5

        # String with currency symbol
        json_obj = {"amount": "$20.5", "currency": "USD"}
        assert parse_amount(json_obj) == 20.5

        # Missing amount
        json_obj = {"currency": "USD"}
        assert parse_amount(json_obj) is None

        # None input
        assert parse_amount(None) is None

    def test_post_process_output(self):
        """Test complete post-processing pipeline."""
        # Clean output
        output = '{"amount": 20, "currency": "USD"}'
        json_str, json_obj, is_valid, amount = post_process_output(output)
        assert json_str == output
        assert json_obj["amount"] == 20
        assert is_valid is True
        assert amount == 20.0

        # Messy output
        output = 'The result is: {"amount": 20.5, "currency": "USD"}'
        json_str, json_obj, is_valid, amount = post_process_output(output)
        assert json_str == '{"amount": 20.5, "currency": "USD"}'
        assert json_obj["amount"] == 20.5
        assert is_valid is True
        assert amount == 20.5

        # Invalid output
        output = "Not a JSON"
        json_str, json_obj, is_valid, amount = post_process_output(output)
        assert json_str == output
        assert json_obj is None
        assert is_valid is False
        assert amount is None


@pytest.mark.integration
class TestInferencePipeline:
    """Integration tests for the complete inference pipeline."""

    @patch("src.inference.load_model")
    @patch("src.inference.run_model_inference")
    def test_inference_pipeline(self, mock_run_inference, mock_load_model):
        """Test the complete inference pipeline."""
        # Mock the model loading and inference
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_metadata = MagicMock()
        mock_load_model.return_value = (mock_model, mock_tokenizer, mock_metadata)
        mock_run_inference.return_value = '{"amount": 25.10, "currency": "USD"}'

        # Run the pipeline
        result = inference_pipeline("twenty-five dollars and ten cents", "model/path")

        # Check the result
        json_str, json_obj, is_valid, amount, raw_output = result
        assert json_str == '{"amount": 25.10, "currency": "USD"}'
        assert json_obj["amount"] == 25.10
        assert json_obj["currency"] == "USD"
        assert is_valid is True
        assert amount == 25.10
        assert raw_output == '{"amount": 25.10, "currency": "USD"}'

        # Check that the functions were called correctly
        mock_load_model.assert_called_once_with("model/path")
        mock_run_inference.assert_called_once()
