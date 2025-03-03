"""
Tests for the inference pipeline module.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.inference import (
    inference_pipeline,
    prepare_batch_inputs,
    prepare_input,
    run_model_inference,
)


class TestInputPreparation:
    """Tests for input preparation functions."""

    def test_prepare_input(self):
        """Test that input text is properly normalized."""
        # Test basic normalization
        assert prepare_input("TWENTY dollars", add_instruction_prefix=False) == "twenty dollars"

        # Test special character removal
        assert prepare_input("$25.50!", add_instruction_prefix=False) == "$25.50"

        # Test whitespace normalization
        assert prepare_input("  five   hundred  dollars  ", add_instruction_prefix=False) == "five hundred dollars"

        # Test currency symbol preservation
        assert prepare_input("€100", add_instruction_prefix=False) == "€100"
        assert prepare_input("$50.25", add_instruction_prefix=False) == "$50.25"
        assert prepare_input("£10.99", add_instruction_prefix=False) == "£10.99"

    def test_prepare_batch_inputs(self):
        """Test that batch inputs are properly processed."""
        inputs = ["$20", "thirty euros", "ONE HUNDRED YEN"]
        expected = ["$20", "thirty euros", "one hundred yen"]

        assert prepare_batch_inputs(inputs, add_instruction_prefix=False) == expected


class TestModelInference:
    """Tests for model inference functions."""

    @patch("src.inference.generate_text")
    def test_run_model_inference_greedy(self, mock_generate_text):
        """Test greedy decoding inference."""
        # Mock the generate_text function
        mock_generate_text.return_value = "20|00"

        # Mock model and tokenizer
        model = MagicMock()
        tokenizer = MagicMock()

        # Run inference with greedy decoding
        result = run_model_inference(model, tokenizer, "twenty dollars", use_greedy_decoding=True)

        # Check the result
        assert result == "20|00"

        # Check that generate_text was called with the right parameters
        mock_generate_text.assert_called_once()
        args, kwargs = mock_generate_text.call_args
        assert kwargs["num_beams"] == 1  # Greedy decoding
        assert kwargs["temperature"] == 1.0

    @patch("src.inference.generate_text")
    def test_run_model_inference_beam_search(self, mock_generate_text):
        """Test beam search inference."""
        # Mock the generate_text function
        mock_generate_text.return_value = "20|00"

        # Mock model and tokenizer
        model = MagicMock()
        tokenizer = MagicMock()

        # Run inference with beam search
        result = run_model_inference(model, tokenizer, "twenty dollars", use_greedy_decoding=False)

        # Check the result
        assert result == "20|00"

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
        mock_run_inference.return_value = "25|10"

        # Run the pipeline
        result = inference_pipeline("twenty-five dollars and ten cents", "model/path")

        # Check the result
        assert result == "25|10"

        # Check that the functions were called correctly
        mock_load_model.assert_called_once_with("model/path")
        mock_run_inference.assert_called_once()
