"""
Model configuration module for FLAN-T5.
"""
import os
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import config


def initialize_model(model_name=None, device=None):
    """
    Initialize the FLAN-T5 model for sequence-to-sequence generation.
    
    Args:
        model_name (str): Name of the pretrained model
        device (str or torch.device): Device to place the model on
        
    Returns:
        transformers.PreTrainedModel: Initialized model
    """
    if model_name is None:
        model_name = config.MODEL_NAME
        
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load the model
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # Move the model to the device
    model = model.to(device)
    
    return model


def save_model(model, tokenizer, output_dir, metadata=None):
    """
    Save the model and tokenizer.
    
    Args:
        model (transformers.PreTrainedModel): Model to save
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer to save
        output_dir (str or Path): Directory to save the model and tokenizer
        metadata (dict): Additional metadata to save
        
    Returns:
        str: Path to the saved model
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the model
    model.save_pretrained(output_dir)
    
    # Save the tokenizer
    tokenizer.save_pretrained(output_dir)
    
    # Save metadata if provided
    if metadata is not None:
        import json
        with open(os.path.join(output_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
    
    return output_dir


def load_model(model_path, device=None):
    """
    Load the model and tokenizer from a saved checkpoint.
    
    Args:
        model_path (str or Path): Path to the saved model
        device (str or torch.device): Device to place the model on
        
    Returns:
        tuple: Model and tokenizer
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load the model
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model = model.to(device)
    
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    return model, tokenizer


def generate_text(model, tokenizer, input_text, max_new_tokens=None, num_beams=None):
    """
    Generate text using the model.
    
    Args:
        model (transformers.PreTrainedModel): Model to use for generation
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer
        input_text (str): Input text to generate from
        max_new_tokens (int): Maximum number of tokens to generate
        num_beams (int): Number of beams for beam search
        
    Returns:
        str: Generated text
    """
    if max_new_tokens is None:
        max_new_tokens = config.MAX_NEW_TOKENS
        
    if num_beams is None:
        num_beams = config.NUM_BEAMS
    
    # Prepare the input
    inputs = tokenizer(input_text, return_tensors="pt")
    
    # Move the inputs to the same device as the model
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
    )
    
    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return generated_text 