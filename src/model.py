"""
Model configuration module for FLAN-T5-Small.
"""

import logging
import os
from typing import Dict, List, Optional, Union

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    T5ForConditionalGeneration,
)

import config

logger = logging.getLogger(__name__)


def initialize_model(model_name: Optional[str] = None, device: Optional[Union[str, torch.device]] = None) -> T5ForConditionalGeneration:
    """
    Initialize the FLAN-T5-Small model for sequence-to-sequence generation.

    Args:
        model_name (str, optional): Name of the pretrained model. Defaults to config.MODEL_NAME.
        device (str or torch.device, optional): Device to place the model on. Defaults to config.DEVICE.

    Returns:
        T5ForConditionalGeneration: Initialized FLAN-T5-Small model
    """
    if model_name is None:
        model_name = config.MODEL_NAME

    if model_name != "google/flan-t5-small":
        logger.warning(f"Expected 'google/flan-t5-small' but got '{model_name}'. Proceeding with {model_name}.")

    if device is None:
        device = config.DEVICE

    logger.info(f"Initializing FLAN-T5-Small model on {device}")

    # Load the model
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Move the model to the device
    model = model.to(device)

    # Set model to evaluation mode
    model.eval()

    return model


def save_model(
    model: T5ForConditionalGeneration,
    tokenizer: AutoTokenizer,
    output_dir: str,
    metadata: Optional[Dict] = None,
) -> str:
    """
    Save the FLAN-T5-Small model and tokenizer with configuration details.

    Args:
        model (T5ForConditionalGeneration): Model to save
        tokenizer (AutoTokenizer): Tokenizer to save
        output_dir (str): Directory to save the model and tokenizer
        metadata (dict, optional): Additional metadata to save. Defaults to None.

    Returns:
        str: Path to the saved model
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Saving model to {output_dir}")

    # Save the model
    model.save_pretrained(output_dir)

    # Save the tokenizer
    tokenizer.save_pretrained(output_dir)

    # Create default metadata if not provided
    if metadata is None:
        metadata = {}

    # Add model configuration details
    metadata.update(
        {
            "model_name": model.config.name_or_path,
            "model_type": model.config.model_type,
            "vocab_size": model.config.vocab_size,
            "hidden_size": model.config.hidden_size,
            "num_layers": model.config.num_decoder_layers,
            "num_heads": model.config.num_heads,
            "device": str(next(model.parameters()).device),
            "max_length": config.MAX_INPUT_LENGTH,
            "max_new_tokens": config.MAX_NEW_TOKENS,
        }
    )

    # Save metadata
    import json

    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    return output_dir


def load_model(model_path: str, device: Optional[Union[str, torch.device]] = None) -> tuple:
    """
    Load the FLAN-T5-Small model and tokenizer from a saved checkpoint.

    Args:
        model_path (str): Path to the saved model
        device (str or torch.device, optional): Device to place the model on. Defaults to config.DEVICE.

    Returns:
        tuple: (model, tokenizer, config)
    """
    if device is None:
        device = config.DEVICE

    logger.info(f"Loading model from {model_path} to {device}")

    # Load the model
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model = model.to(device)
    model.eval()

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load metadata if available
    metadata = None
    metadata_path = os.path.join(model_path, "metadata.json")
    if os.path.exists(metadata_path):
        import json

        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        logger.info(f"Loaded model metadata: {metadata}")

    # Verify model is ready for use
    if model is None or tokenizer is None:
        raise ValueError("Failed to load model or tokenizer")

    logger.info(f"Model loaded successfully. Device: {next(model.parameters()).device}")

    return model, tokenizer, metadata


def prepare_inputs(
    tokenizer: AutoTokenizer,
    text: Union[str, List[str]],
    max_length: Optional[int] = None,
    return_tensors: str = "pt",
) -> Dict[str, torch.Tensor]:
    """
    Prepare inputs for inference by tokenizing and formatting.

    Args:
        tokenizer (AutoTokenizer): Tokenizer for the model
        text (str or List[str]): Input text or batch of texts
        max_length (int, optional): Maximum input length. Defaults to config.MAX_INPUT_LENGTH.
        return_tensors (str, optional): Return tensor type. Defaults to "pt" (PyTorch).

    Returns:
        Dict[str, torch.Tensor]: Tokenized inputs ready for the model
    """
    if max_length is None:
        max_length = config.MAX_INPUT_LENGTH

    # Tokenize the input text
    inputs = tokenizer(
        text,
        padding="max_length" if isinstance(text, list) else False,
        truncation=True,
        max_length=max_length,
        return_tensors=return_tensors,
    )

    return inputs


def batch_process(
    model: T5ForConditionalGeneration,
    tokenizer: AutoTokenizer,
    texts: List[str],
    max_length: Optional[int] = None,
    max_new_tokens: Optional[int] = None,
    num_beams: Optional[int] = None,
    temperature: Optional[float] = None,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
) -> List[str]:
    """
    Process a batch of input texts for inference.

    Args:
        model (T5ForConditionalGeneration): The FLAN-T5-Small model
        tokenizer (AutoTokenizer): Tokenizer for the model
        texts (List[str]): List of input texts to process
        max_length (int, optional): Maximum input length. Defaults to config.MAX_INPUT_LENGTH.
        max_new_tokens (int, optional): Maximum number of tokens to generate. Defaults to config.MAX_NEW_TOKENS.
        num_beams (int, optional): Number of beams for beam search. Defaults to 4.
        temperature (float, optional): Sampling temperature. Defaults to 1.0.
        top_k (int, optional): Top-k sampling. Defaults to 50.
        top_p (float, optional): Top-p sampling. Defaults to 1.0.

    Returns:
        List[str]: List of generated texts
    """
    if max_length is None:
        max_length = config.MAX_INPUT_LENGTH

    if max_new_tokens is None:
        max_new_tokens = config.MAX_NEW_TOKENS

    if num_beams is None:
        num_beams = 4  # Use beam search by default for better quality

    if temperature is None:
        temperature = 1.0

    if top_k is None:
        top_k = 50

    if top_p is None:
        top_p = 1.0

    # Prepare batch inputs
    batch_inputs = prepare_inputs(tokenizer, texts, max_length)

    # Move inputs to the same device as the model
    device = next(model.parameters()).device
    batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}

    # Configure generation parameters
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "num_beams": num_beams,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "repetition_penalty": 1.2,  # Increased to reduce repetition
    }

    # Add beam search parameters if using beam search
    if num_beams > 1:
        gen_kwargs.update(
            {
                "early_stopping": True,
                "length_penalty": 1.0,
                "no_repeat_ngram_size": 2,  # Prevent repeating bigrams
            }
        )

    # Add sampling parameters if using sampling
    if temperature != 1.0:
        gen_kwargs.update(
            {
                "do_sample": True,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
            }
        )

    # Generate text
    with torch.no_grad():
        outputs = model.generate(**batch_inputs, **gen_kwargs)

    # Decode outputs to text
    generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return generated_texts


def generate_text(
    model: T5ForConditionalGeneration,
    tokenizer: AutoTokenizer,
    input_text: str,
    max_length: Optional[int] = None,
    max_new_tokens: Optional[int] = None,
    num_beams: Optional[int] = None,
    temperature: Optional[float] = None,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
) -> str:
    """
    Generate text using the FLAN-T5-Small model.

    Args:
        model (T5ForConditionalGeneration): Model to use for generation
        tokenizer (AutoTokenizer): Tokenizer for the model
        input_text (str): Input text to generate from
        max_length (int, optional): Maximum input length. Defaults to config.MAX_INPUT_LENGTH.
        max_new_tokens (int, optional): Maximum number of tokens to generate. Defaults to config.MAX_NEW_TOKENS.
        num_beams (int, optional): Number of beams for beam search. Defaults to 4.
        temperature (float, optional): Sampling temperature. Defaults to 1.0.
        top_k (int, optional): Top-k sampling. Defaults to 50.
        top_p (float, optional): Top-p sampling. Defaults to 1.0.

    Returns:
        str: Generated text
    """
    if max_length is None:
        max_length = config.MAX_INPUT_LENGTH

    if max_new_tokens is None:
        max_new_tokens = config.MAX_NEW_TOKENS

    if num_beams is None:
        num_beams = 4  # Use beam search by default for better quality

    if temperature is None:
        temperature = 1.0

    if top_k is None:
        top_k = 50

    if top_p is None:
        top_p = 1.0

    # Prepare the input
    inputs = prepare_inputs(tokenizer, input_text, max_length)

    # Move the inputs to the same device as the model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Configure generation parameters
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "num_beams": num_beams,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "repetition_penalty": 1.2,  # Increased to reduce repetition
    }

    # Add beam search parameters if using beam search
    if num_beams > 1:
        gen_kwargs.update(
            {
                "early_stopping": True,
                "length_penalty": 1.0,
                "no_repeat_ngram_size": 2,  # Prevent repeating bigrams
            }
        )

    # Add sampling parameters if using sampling
    if temperature != 1.0:
        gen_kwargs.update(
            {
                "do_sample": True,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
            }
        )

    # Generate
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return generated_text


def test_model():
    """
    Test function to verify model configuration and basic inference.

    Returns:
        tuple: (model, tokenizer, test_output)
    """
    logger.info("Testing FLAN-T5-Small model configuration")

    # Initialize model and tokenizer
    model = initialize_model()
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

    # Test input
    test_input = "Translate English to French: Hello, how are you?"

    # Generate prediction
    logger.info(f"Test input: {test_input}")
    output = generate_text(model, tokenizer, test_input)
    logger.info(f"Model output: {output}")

    # Test batch processing
    batch_inputs = [
        "Translate English to French: Hello world",
        "Summarize: The quick brown fox jumps over the lazy dog. The dog was not impressed.",
    ]
    batch_outputs = batch_process(model, tokenizer, batch_inputs)
    logger.info(f"Batch outputs: {batch_outputs}")

    return model, tokenizer, output


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Test the model configuration
    test_model()
