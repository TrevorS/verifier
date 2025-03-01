"""
Tests for the training pipeline.
"""

import json
import logging
import os
import tempfile
from pathlib import Path

import pytest

import config
from src.trainer import (
    configure_training_args,
    is_valid_json,
    setup_trainer,
    train_model,
)

# Configure logging for tests
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@pytest.fixture
def sample_data():
    """Create temporary sample data files for training tests."""
    examples = [
        ("twenty-five dollars and ten cents", {"amount": 25.10, "currency": "USD"}),
        ("five dollars", {"amount": 5.00, "currency": "USD"}),
        ("one hundred euros", {"amount": 100.00, "currency": "EUR"}),
        ("seventy-five cents", {"amount": 0.75, "currency": "USD"}),
        ("two thousand three hundred yen", {"amount": 2300.00, "currency": "JPY"}),
        ("fifty british pounds", {"amount": 50.00, "currency": "GBP"}),
        ("twelve dollars and fifty cents", {"amount": 12.50, "currency": "USD"}),
        ("one million dollars", {"amount": 1000000.00, "currency": "USD"}),
        ("three euros and twenty cents", {"amount": 3.20, "currency": "EUR"}),
        ("seven hundred fifty canadian dollars", {"amount": 750.00, "currency": "CAD"}),
    ]

    num_examples = 20
    train_size = int(num_examples * 0.7)
    val_size = num_examples - train_size

    train_data = []
    val_data = []

    # Generate training data
    for i in range(train_size):
        example_idx = i % len(examples)
        train_data.append({"input": examples[example_idx][0], "output": json.dumps(examples[example_idx][1])})

    # Generate validation data
    for i in range(val_size):
        example_idx = i % len(examples)
        val_data.append({"input": examples[example_idx][0], "output": json.dumps(examples[example_idx][1])})

    with tempfile.TemporaryDirectory() as temp_dir:
        train_path = Path(temp_dir) / "train_sample.jsonl"
        val_path = Path(temp_dir) / "val_sample.jsonl"
        output_dir = Path(temp_dir) / "model_outputs"
        os.makedirs(output_dir, exist_ok=True)

        with open(train_path, "w") as f:
            for item in train_data:
                f.write(json.dumps(item) + "\n")

        with open(val_path, "w") as f:
            for item in val_data:
                f.write(json.dumps(item) + "\n")

        logger.info(f"Created sample training data: {train_path} ({len(train_data)} examples)")
        logger.info(f"Created sample validation data: {val_path} ({len(val_data)} examples)")

        yield {"train_path": train_path, "val_path": val_path, "output_dir": output_dir, "temp_dir": temp_dir}


def test_is_valid_json():
    """Test the JSON validation function."""
    # Test valid JSON
    assert is_valid_json('{"amount": 25.10, "currency": "USD"}') is True

    # Test invalid JSON
    assert is_valid_json('{"amount": 25.10, "currency": "USD"') is False
    assert is_valid_json("not json at all") is False
    assert is_valid_json("") is False


def test_configure_training_args():
    """Test training arguments configuration."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test with default arguments
        args = configure_training_args(temp_dir)

        # Verify basic properties
        assert args.output_dir == temp_dir
        assert args.num_train_epochs == config.NUM_EPOCHS
        assert args.per_device_train_batch_size == config.BATCH_SIZE
        assert args.learning_rate == config.LEARNING_RATE

        # Test with custom arguments
        custom_args = configure_training_args(temp_dir, num_train_epochs=5, batch_size=32, learning_rate=5e-5, early_stopping_patience=2)

        assert custom_args.num_train_epochs == 5
        assert custom_args.per_device_train_batch_size == 32
        assert custom_args.learning_rate == 5e-5


@pytest.mark.slow
def test_training_pipeline_end_to_end(sample_data):
    """Test the complete training pipeline end-to-end."""
    # Skip this test by default unless explicitly requested
    pytest.skip("Skipping end-to-end training test unless explicitly requested. Run with --run-slow to enable.")

    try:
        best_model_path = train_model(
            model_name=config.MODEL_NAME,
            train_data_path=sample_data["train_path"],
            val_data_path=sample_data["val_path"],
            output_dir=sample_data["output_dir"],
            wandb_logging=False,  # Disable W&B logging for tests
        )

        # Check that the model was saved
        assert os.path.exists(best_model_path)
        assert os.path.isdir(best_model_path)

        # Check that the model contains the expected files
        assert os.path.exists(os.path.join(best_model_path, "config.json"))
        assert os.path.exists(os.path.join(best_model_path, "pytorch_model.bin"))
        assert os.path.exists(os.path.join(best_model_path, "tokenizer.json"))

        # Check that reports were generated
        reports_dir = os.path.join(sample_data["output_dir"], "reports")
        assert os.path.exists(reports_dir)
        assert os.path.exists(os.path.join(reports_dir, "prediction_examples.json"))
        assert os.path.exists(os.path.join(reports_dir, "prediction_summary.json"))

    except Exception as e:
        pytest.fail(f"Training pipeline test failed with error: {str(e)}")


@pytest.mark.integration
def test_trainer_setup(sample_data):
    """Test the trainer setup without full training."""
    from src.dataset import prepare_dataset
    from src.model import initialize_model

    # Prepare minimal datasets
    preprocessed_dataset, tokenizer, data_collator = prepare_dataset(
        sample_data["train_path"],
        sample_data["val_path"],
        config.MODEL_NAME,
    )

    # Initialize model (possibly with smaller config for tests)
    model = initialize_model(config.MODEL_NAME)

    # Set up trainer
    trainer = setup_trainer(
        model,
        tokenizer,
        preprocessed_dataset["train"],
        preprocessed_dataset["validation"],
        data_collator,
        sample_data["output_dir"],
    )

    # Check trainer configuration
    assert trainer.model is not None
    assert trainer.args is not None
    assert trainer.train_dataset is not None
    assert trainer.eval_dataset is not None
    assert trainer.tokenizer is not None
    assert trainer.data_collator is not None
    assert trainer.compute_metrics is not None

    # Check tokenizer attachment
    assert hasattr(trainer.model, "tokenizer")
    assert trainer.model.tokenizer is tokenizer
