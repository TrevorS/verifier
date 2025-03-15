#!/usr/bin/env python

import argparse
import json
from pathlib import Path

from datasets import Dataset, DatasetDict


def load_jsonl(file_path: str) -> list[dict]:
    """Load data from a JSONL file."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def load_data(data_dir: str) -> DatasetDict:
    """Load data from the data directory and create a DatasetDict."""
    data_dir = Path(data_dir)

    # Check if all required files exist
    required_files = ["train.jsonl", "val.jsonl", "test.jsonl"]
    for file in required_files:
        if not (data_dir / file).exists():
            raise FileNotFoundError(f"Required file {file} not found in {data_dir}")

    # Load data from each file
    datasets = {}
    for split in ["train", "val", "test"]:
        file_path = data_dir / f"{split}.jsonl"
        print(f"Loading {file_path}...")
        data = load_jsonl(str(file_path))
        datasets[split] = Dataset.from_list(data)

    return DatasetDict(datasets)


def upload_to_hf(
    dataset: DatasetDict,
    repo_id: str,
) -> None:
    # Push to hub
    print(f"Uploading dataset to {repo_id}...")
    dataset.push_to_hub(repo_id)
    print(f"Dataset uploaded successfully to https://huggingface.co/datasets/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Load data and upload to Hugging Face Hub")
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="Repository ID on Hugging Face Hub (e.g., 'username/dataset-name')",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Directory containing the data files (default: 'data')",
    )

    args = parser.parse_args()

    # Load data
    dataset = load_data(args.data_dir)

    # Print dataset info
    print("\nDataset info:")
    for split, ds in dataset.items():
        print(f"  {split}: {len(ds)} examples")

    # Upload to Hugging Face Hub
    upload_to_hf(
        dataset=dataset,
        repo_id=args.repo_id,
    )


if __name__ == "__main__":
    main()
