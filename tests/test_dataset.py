"""
Tests for the dataset module.
"""

import json
import tempfile
from pathlib import Path

import datasets
import pytest
from transformers import PreTrainedTokenizerBase

import config
from src.dataset import (
    configure_tokenizer,
    load_dataset,
    prepare_dataset,
    preprocess_dataset,
)


@pytest.fixture
def sample_data():
    """Create temporary sample data files."""
    train_data = [
        {
            "input": "one hundred twenty-three dollars and forty-five cents", 
            "target": {"dollars": 123.0, "cents": 45.0}
        },
        {
            "input": "five dollars", 
            "target": {"dollars": 5.0, "cents": 0.0}
        },
        {
            "input": "seventy-five cents", 
            "target": {"dollars": 0.0, "cents": 75.0}
        },
    ]

    val_data = [
        {
            "input": "two thousand dollars", 
            "target": {"dollars": 2000.0, "cents": 0.0}
        },
        {
            "input": "one dollar and one cent", 
            "target": {"dollars": 1.0, "cents": 1.0}
        },
    ]

    with tempfile.TemporaryDirectory() as temp_dir:
        train_path = Path(temp_dir) / "train.jsonl"
        val_path = Path(temp_dir) / "val.jsonl"

        with open(train_path, "w") as f:
            for item in train_data:
                f.write(json.dumps(item) + "\n")

        with open(val_path, "w") as f:
            for item in val_data:
                f.write(json.dumps(item) + "\n")

        yield train_path, val_path


def test_load_dataset(sample_data):
    """Test loading JSONL data."""
    train_path, val_path = sample_data

    # Test with both train and validation
    dataset = load_dataset(train_path, val_path)
    assert isinstance(dataset, datasets.DatasetDict)
    assert "train" in dataset
    assert "validation" in dataset
    assert len(dataset["train"]) == 3
    assert len(dataset["validation"]) == 2

    # Test with only train
    train_only = load_dataset(train_path)
    assert isinstance(train_only, datasets.DatasetDict)
    assert "train" in train_only
    assert "validation" not in train_only
    assert len(train_only["train"]) == 3


def test_configure_tokenizer():
    """Test tokenizer configuration."""
    model_name = config.MODEL_NAME
    tokenizer = configure_tokenizer(model_name)

    # Verify we got a tokenizer
    assert tokenizer is not None
    assert isinstance(tokenizer, PreTrainedTokenizerBase)

    # Test tokenization
    input_text = "one hundred dollars"
    tokens = tokenizer(input_text)
    assert len(tokens["input_ids"]) > 0


def test_preprocess_dataset(sample_data):
    """Test dataset preprocessing."""
    train_path, val_path = sample_data
    dataset = load_dataset(train_path, val_path)
    tokenizer = configure_tokenizer(config.MODEL_NAME)

    # Test preprocessing
    processed = preprocess_dataset(dataset, tokenizer)

    # Verify the dataset structure
    assert "train" in processed
    assert "validation" in processed

    # Verify tokenized format
    assert "input_ids" in processed["train"].features
    assert "attention_mask" in processed["train"].features
    assert "labels" in processed["train"].features

    # Verify data was properly processed
    assert len(processed["train"]) == len(dataset["train"])
    assert len(processed["validation"]) == len(dataset["validation"])


def test_prepare_dataset(sample_data):
    """Test the complete dataset preparation pipeline."""
    train_path, val_path = sample_data

    # Test the full pipeline
    dataset, tokenizer, data_collator = prepare_dataset(train_path, val_path, config.MODEL_NAME)

    # Verify we got all components
    assert dataset is not None
    assert tokenizer is not None
    assert data_collator is not None

    # Verify dataset structure
    assert "train" in dataset
    assert "validation" in dataset
    assert "input_ids" in dataset["train"].features
    assert "attention_mask" in dataset["train"].features
    assert "labels" in dataset["train"].features


def test_dataset_performance(sample_data):
    """Test dataset loading speed and batch processing."""
    train_path, val_path = sample_data

    # Measure loading time
    import time

    start_time = time.time()
    dataset = load_dataset(train_path, val_path)
    load_time = time.time() - start_time

    # Just a basic check that loading happened in a reasonable time
    assert load_time < 5.0, f"Dataset loading took too long: {load_time:.2f}s"

    # Test batch processing
    tokenizer = configure_tokenizer(config.MODEL_NAME)

    start_time = time.time()
    preprocess_dataset(dataset, tokenizer)
    process_time = time.time() - start_time

    # Batched processing should be reasonably fast
    assert process_time < 5.0, f"Dataset preprocessing took too long: {process_time:.2f}s"
