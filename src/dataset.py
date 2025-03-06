"""
Dataset preparation module using HuggingFace datasets.
"""

import random

import datasets
from transformers import AutoTokenizer, DataCollatorForSeq2Seq

import config
from src.prompts import create_prompt


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


def select_examples(dataset: datasets.Dataset, num_examples: int = 2) -> list[dict[str, str]]:
    """
    Select examples from the dataset for few-shot learning.

    Args:
        dataset: The dataset to select examples from
        num_examples: Number of examples to select

    Returns:
        list[dict[str, str]]: Selected examples
    """
    indices = random.sample(range(len(dataset)), num_examples)
    examples = []
    for idx in indices:
        examples.append(
            {
                "input": dataset[idx]["input"],
                "target": dataset[idx]["target"],
            }
        )
    return examples


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
        outputs = examples["target"]

        # Select examples for few-shot learning if enabled
        if config.USE_EXAMPLES:
            # Get examples from the training set
            train_examples = select_examples(dataset["train"], num_examples=config.NUM_EXAMPLES)
        else:
            train_examples = None

        # Create prompts with examples if enabled
        processed_inputs = [create_prompt(text, examples=train_examples, instruction_prefix=config.INSTRUCTION_PREFIX) for text in inputs]

        # Tokenize inputs with instruction prefix and examples
        model_inputs = tokenizer(
            processed_inputs,
            max_length=max_input_length,
            padding="max_length",
            truncation=True,
        )

        # Tokenize targets
        target_encoding = tokenizer(
            text_target=outputs,
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
