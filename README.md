# Monetary Expressions to JSON Converter

A sequence-to-sequence model that converts verbal monetary expressions to JSON. The model transforms phrases like "twenty-five dollars and ten cents" into structured JSON like `{"amount": 25.10}`.

## Project Structure

```
.
├── config.py             # Configuration settings
├── main.py               # Entry point script
├── data/                 # Dataset directory
│   ├── train.jsonl       # Training data
│   ├── val.jsonl         # Validation data
│   └── test.jsonl        # Test data
├── models/               # Saved model checkpoints
├── src/                  # Source code
```

## Requirements

- Python 3.12
- Dependencies listed in `requirements.txt`

## Installation

1. Create a Python 3.12 virtual environment:

```bash
python3.12 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies using uv:

```bash
pip install uv
uv pip install -r requirements.txt
```

## Usage

### Generating Synthetic Data

Generate synthetic training and validation data:

```bash
python main.py generate-data --num-examples 10000
```

### Training the Model

Train the model using generated data:

```bash
python main.py train
```

To disable Weights & Biases logging:

```bash
python main.py train --no-wandb
```

### Evaluation

Evaluate the model on the test set:

```bash
python main.py evaluate --model-path models/checkpoints/best
```

### Inference

Run inference on a single expression:

```bash
python main.py infer --model-path models/checkpoints/best --text "twenty-five dollars and ten cents"
```

Or process multiple expressions from a file:

```bash
python main.py infer --model-path models/checkpoints/best --input-file examples.txt
```

## Implementation Details

This project uses the FLAN-T5-Small model from Google as the base model for the sequence-to-sequence task of converting monetary expressions to JSON.

The model is fine-tuned on a synthetic dataset of verbal monetary expressions paired with their corresponding JSON representations.

## License

[MIT](LICENSE)
