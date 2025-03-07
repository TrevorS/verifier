#!/usr/bin/env python3
"""
Predict whether a verbal amount matches a decimal amount using the trained model.
"""

import argparse
import json
import sys
from datetime import datetime

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_model_and_tokenizer(model_path):
    """Load the trained model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    model.eval()  # Set model to evaluation mode
    return model, tokenizer, device


def predict(model, tokenizer, verbal_amount, decimal_amount):
    """Make a prediction for the given input pair."""
    # Tokenize inputs with same settings as training
    print(f"Tokenizing inputs: {verbal_amount} and {decimal_amount}")
    inputs = tokenizer(
        verbal_amount,
        decimal_amount,
        padding="max_length",  # Match training settings
        truncation=True,
        max_length=128,  # Match training settings
        return_tensors="pt",
    )

    # Move inputs to same device as model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][prediction].item()
    return {
        "timestamp": datetime.now().isoformat(),
        "input": {"verbal_amount": verbal_amount, "decimal_amount": decimal_amount},
        "prediction": {
            "match": bool(prediction),
            "confidence": round(confidence * 100, 2),
            "probabilities": {"no_match": round(float(probabilities[0][0]), 4), "match": round(float(probabilities[0][1]), 4)},
        },
        "model_info": {"name": model.config.name_or_path, "type": model.config.model_type},
    }


def main():
    parser = argparse.ArgumentParser(description="Verify if verbal and decimal amounts match.")
    parser.add_argument("verbal_amount", help="The verbal representation of the amount")
    parser.add_argument("decimal_amount", help="The decimal representation of the amount")
    parser.add_argument("--model-path", help="Path to the trained model")
    args = parser.parse_args()

    try:
        if args.model_path is None:
            print("No model path provided, exiting")
            sys.exit(1)

        print(f"Loading model from {args.model_path}")
        model, tokenizer, device = load_model_and_tokenizer(args.model_path)
        print(f"Model loaded from {args.model_path} on {device}")
        result = predict(model, tokenizer, args.verbal_amount, args.decimal_amount)
        json.dump(result, sys.stdout, indent=2)
        print()  # Add newline after JSON output
    except Exception as e:
        json.dump({"error": str(e), "timestamp": datetime.now().isoformat()}, sys.stdout, indent=2)
        print()
        sys.exit(1)


if __name__ == "__main__":
    main()
