#!/usr/bin/env python
"""
Tests for the FLAN-T5-Small model implementation.
"""

import logging
import os

import pytest
import torch
from transformers import AutoTokenizer

import config
from src.model import (
    batch_process,
    generate_text,
    initialize_model,
    load_model,
    prepare_inputs,
    save_model,
)

# Configure logging for tests
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Fixture for model and tokenizer to be reused across tests
@pytest.fixture(scope="module")
def model_and_tokenizer():
    """Fixture that provides the model and tokenizer for tests."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    logger.info("Initializing FLAN-T5-Small model and tokenizer...")
    model = initialize_model(config.MODEL_NAME, device)
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

    return model, tokenizer, device


class TestModelInitialization:
    """Tests for model initialization."""

    def test_model_initialization(self, model_and_tokenizer):
        """Test that the model initializes correctly."""
        model, tokenizer, _ = model_and_tokenizer

        # Verify model and tokenizer are not None
        assert model is not None
        assert tokenizer is not None

        # Verify model is in evaluation mode
        assert not model.training

        # Verify model architecture
        assert model.config.model_type == "t5"


class TestTextGeneration:
    """Tests for text generation functionality."""

    def test_single_input_generation(self, model_and_tokenizer):
        """Test generating text from a single input."""
        model, tokenizer, _ = model_and_tokenizer

        input_text = "Translate to French: Hello, how are you?"
        output = generate_text(model, tokenizer, input_text, max_new_tokens=50)

        # Verify output is a non-empty string
        assert isinstance(output, str)
        assert len(output) > 0
        logger.info(f"Input: {input_text}")
        logger.info(f"Output: {output}")

    def test_batch_processing(self, model_and_tokenizer):
        """Test batch processing of multiple inputs."""
        model, tokenizer, _ = model_and_tokenizer

        test_inputs = [
            "Translate to French: Hello, how are you?",
            "Summarize: The FLAN-T5 model is a version of T5 that has been finetuned on a mixture of tasks.",  # noqa: E501
            "What is the capital of France?",
        ]

        batch_outputs = batch_process(model, tokenizer, test_inputs, max_new_tokens=50)

        # Verify correct number of outputs
        assert len(batch_outputs) == len(test_inputs)

        # Verify all outputs are non-empty strings
        for output in batch_outputs:
            assert isinstance(output, str)
            assert len(output) > 0

    def test_classification_task(self, model_and_tokenizer):
        """Test a classification task."""
        model, tokenizer, _ = model_and_tokenizer

        input_text = "Classify this text as positive or negative: I love this product!"
        output = generate_text(model, tokenizer, input_text, max_new_tokens=20)

        # Expecting output to be either "positive" or "negative"
        assert output.lower().strip() in ["positive", "negative"]
        logger.info(f"Classification result: {output}")


class TestModelSaveLoad:
    """Tests for model saving and loading functionality."""

    def test_save_and_load(self, model_and_tokenizer):
        """Test saving and loading the model."""
        model, tokenizer, device = model_and_tokenizer

        # Create a test save path
        save_path = "models/flan-t5-small-test"

        # Add test metadata
        metadata = {"test_info": "This is a test checkpoint", "timestamp": "2023-06-01"}

        # Save the model
        saved_path = save_model(model, tokenizer, save_path, metadata)
        assert os.path.exists(saved_path)

        # Load the model
        loaded_model, loaded_tokenizer, loaded_metadata = load_model(saved_path, device)

        # Verify loaded model and tokenizer
        assert loaded_model is not None
        assert loaded_tokenizer is not None
        assert not loaded_model.training

        # Verify metadata was saved and loaded correctly
        assert "test_info" in loaded_metadata
        assert loaded_metadata["test_info"] == "This is a test checkpoint"

        # Test loaded model works
        verification_input = "Translate to Spanish: Good morning!"
        loaded_output = generate_text(loaded_model, loaded_tokenizer, verification_input, max_new_tokens=50)

        # Verify output is a non-empty string
        assert isinstance(loaded_output, str)
        assert len(loaded_output) > 0
        logger.info(f"Loaded model output: {loaded_output}")


class TestInputPreparation:
    """Tests for input preparation functionality."""

    def test_prepare_single_input(self, model_and_tokenizer):
        """Test preparing a single input for inference."""
        _, tokenizer, _ = model_and_tokenizer

        input_text = "Test input"
        prepared_input = prepare_inputs(tokenizer, input_text)

        # Verify input has correct keys and tensor shapes
        assert "input_ids" in prepared_input
        assert "attention_mask" in prepared_input
        assert prepared_input["input_ids"].dim() == 2  # [batch_size=1, seq_len]

    def test_prepare_batch_input(self, model_and_tokenizer):
        """Test preparing a batch of inputs for inference."""
        _, tokenizer, _ = model_and_tokenizer

        batch_inputs = ["Test input 1", "Test input 2", "Test input 3"]
        prepared_batch = prepare_inputs(tokenizer, batch_inputs)

        # Verify batch input has correct shape
        assert prepared_batch["input_ids"].shape[0] == len(batch_inputs)


if __name__ == "__main__":
    pytest.main()
