"""
Dataset preparation module using HuggingFace datasets.
"""

import datasets
from transformers import AutoTokenizer, DataCollatorForSeq2Seq

import config
from src.model import MonetaryAmountDataCollator


def load_dataset(train_path, val_path=None):
    """
    Load datasets from JSONL files.

    Args:
        train_path (str or Path): Path to the training data
        val_path (str or Path): Path to the validation data

    Returns:
        datasets.DatasetDict: Dataset dictionary with train and validation splits
    """
    data_files = {"train": str(train_path)}

    if val_path is not None:
        data_files["validation"] = str(val_path)

    # Load the dataset
    dataset = datasets.load_dataset("json", data_files=data_files)

    return dataset


def configure_tokenizer(model_name):
    """
    Configure the tokenizer for the sequence-to-sequence task.

    Args:
        model_name (str): Name of the pretrained model

    Returns:
        transformers.PreTrainedTokenizer: Configured tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer


def preprocess_dataset(dataset, tokenizer, max_input_length=None, max_target_length=None):
    """
    Preprocess the dataset by tokenizing inputs.
    For regression targets, we keep the target dictionary intact.

    Args:
        dataset (datasets.DatasetDict): Dataset dictionary
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer
        max_input_length (int): Maximum input length
        max_target_length (int): Maximum target length (not used for regression)

    Returns:
        datasets.DatasetDict: Preprocessed dataset
    """
    if max_input_length is None:
        max_input_length = config.MAX_INPUT_LENGTH

    def preprocess_function(examples):
        """Tokenize the inputs and prepare targets for regression."""
        # Get the input texts
        inputs = examples["input"]

        # Add instruction prefix to leverage FLAN-T5's instruction-following capabilities
        instructions = [config.INSTRUCTION_PREFIX + ": " + text for text in inputs]

        # Tokenize inputs with instruction prefix
        model_inputs = tokenizer(
            instructions,
            max_length=max_input_length,
            padding="max_length",
            truncation=True,
        )

        # Process targets for regression
        # Convert targets to the format expected by the model
        # The MonetaryAmountDataCollator will handle converting it to tensors
        targets = []
        for target in examples["target"]:
            # Check if target is already a dictionary with dollars and cents
            if isinstance(target, dict) and "dollars" in target and "cents" in target:
                targets.append([target["dollars"], target["cents"]])
            else:
                # Handle string format like "123|45"
                try:
                    dollars_str, cents_str = target.split("|")
                    dollars = float(dollars_str)
                    cents = float(cents_str)
                    targets.append([dollars, cents])
                except (ValueError, AttributeError):
                    # If target is not in expected format, use default values
                    targets.append([0.0, 0.0])
                    
        # Add the processed targets
        model_inputs["labels"] = targets

        return model_inputs

    # Apply preprocessing to the dataset
    preprocessed_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    # Debug the dataset structure
    import logging
    logger = logging.getLogger(__name__)
    # logger.info(f"Dataset columns after preprocessing: {preprocessed_dataset['train'].column_names}")
    logger.info(f"First example after preprocessing: {preprocessed_dataset['train'][0]}")

    return preprocessed_dataset


def prepare_dataset(train_path, val_path, model_name, max_input_length=None, max_target_length=None):
    """
    Prepare the complete dataset for training.

    Args:
        train_path (str or Path): Path to the training data
        val_path (str or Path): Path to the validation data
        model_name (str): Name of the pretrained model
        max_input_length (int): Maximum input length
        max_target_length (int): Maximum target length

    Returns:
        tuple: Preprocessed dataset, tokenizer, and data collator
    """
    # Load the dataset
    dataset = load_dataset(train_path, val_path)

    # Configure the tokenizer
    tokenizer = configure_tokenizer(model_name)

    # Preprocess the dataset
    preprocessed_dataset = preprocess_dataset(
        dataset,
        tokenizer,
        max_input_length,
        max_target_length,
    )

    # Create the data collator for regression
    data_collator = MonetaryAmountDataCollator(tokenizer)

    return preprocessed_dataset, tokenizer, data_collator
