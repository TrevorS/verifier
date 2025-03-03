"""
Dataset preparation module using HuggingFace datasets.
"""

import datasets
from transformers import AutoTokenizer, DataCollatorForSeq2Seq

import config


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
    Preprocess the dataset by tokenizing inputs and targets.

    Args:
        dataset (datasets.DatasetDict): Dataset dictionary
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer
        max_input_length (int): Maximum input length
        max_target_length (int): Maximum target length

    Returns:
        datasets.DatasetDict: Preprocessed dataset
    """
    if max_input_length is None:
        max_input_length = config.MAX_INPUT_LENGTH

    if max_target_length is None:
        max_target_length = config.MAX_TARGET_LENGTH

    def preprocess_function(examples):
        """Tokenize the inputs and targets."""
        # Get the input texts
        inputs = examples["input"]

        # Add instruction prefix to leverage FLAN-T5's instruction-following capabilities
        instructions = [config.INSTRUCTION_PREFIX + ": " + text for text in inputs]

        # Get the target texts
        targets = examples["target"]

        # Tokenize inputs with instruction prefix
        model_inputs = tokenizer(
            instructions,
            max_length=max_input_length,
            padding="max_length",
            truncation=True,
        )

        target_encoding = tokenizer(
            text_target=targets,
            max_length=max_target_length,
            padding="max_length",
            truncation=True,
        )

        # Add the target token ids as labels
        model_inputs["labels"] = target_encoding["input_ids"]

        return model_inputs

    # Apply preprocessing to the dataset
    preprocessed_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

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

    # Create the data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=None,
        padding="max_length",
        max_length=max(
            max_input_length or config.MAX_INPUT_LENGTH,
            max_target_length or config.MAX_TARGET_LENGTH,
        ),
    )

    return preprocessed_dataset, tokenizer, data_collator
