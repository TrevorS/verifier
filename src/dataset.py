import datasets


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
