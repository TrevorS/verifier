"""
Model configuration module for FLAN-T5-Small.
"""

import logging
import os
from typing import Dict, List, Optional, Union, Tuple

import torch
import torch.nn as nn
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    T5ForConditionalGeneration,
    T5PreTrainedModel,
    T5EncoderModel,
    T5Config,
    Trainer,
    trainer_utils,
)

import config

logger = logging.getLogger(__name__)


class MonetaryAmountModel(T5PreTrainedModel):
    """
    Model for predicting monetary amounts as two separate float values (dollars and cents).
    Uses T5 encoder as the backbone with two regression heads.
    """

    def __init__(self, t5_config):
        super().__init__(t5_config)
        
        # T5 encoder for processing input text
        self.encoder = T5EncoderModel(t5_config)
        
        # Regression heads
        hidden_size = t5_config.hidden_size
        self.dollars_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1)
        )
        
        self.cents_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        dollars_target: Optional[torch.Tensor] = None,
        cents_target: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Forward pass of the model.
        
        Args:
            input_ids: Tensor of input token ids
            attention_mask: Tensor of attention mask
            labels: Optional tensor of shape (batch_size, 2) containing [dollars, cents] targets
            dollars_target: Optional tensor of target dollar values (for direct use)
            cents_target: Optional tensor of target cent values (for direct use)
            return_dict: Whether to return a dictionary (required by Trainer)
            
        Returns:
            During training (if targets provided):
                Dict containing loss and predictions
            During inference:
                Tuple of (dollars_pred, cents_pred) if return_dict is False
                Dict containing predictions if return_dict is True
        """
        # Get encoder outputs
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        # Use [CLS] token representation (first token) for prediction
        sequence_output = encoder_outputs.last_hidden_state[:, 0, :]
        
        # Get predictions from both heads
        dollars_pred = self.dollars_head(sequence_output)
        cents_pred = self.cents_head(sequence_output)
        
        # Squeeze predictions to match target shape and ensure float32
        dollars_pred = dollars_pred.squeeze(-1).to(torch.float32)
        cents_pred = cents_pred.squeeze(-1).to(torch.float32)

        # Handle different ways of providing targets
        if labels is not None:
            dollars_target = labels[:, 0].to(torch.float32)  # First column is dollars
            cents_target = labels[:, 1].to(torch.float32)    # Second column is cents
        
        loss = None
        dollars_loss = None
        cents_loss = None
        if dollars_target is not None and cents_target is not None:
            # Ensure targets are float32
            dollars_target = dollars_target.to(torch.float32)
            cents_target = cents_target.to(torch.float32)
            
            # Calculate MSE loss for both outputs
            dollars_loss = nn.functional.mse_loss(dollars_pred, dollars_target)
            cents_loss = nn.functional.mse_loss(cents_pred, cents_target)
            
            # Total loss is the sum of both losses
            loss = dollars_loss + cents_loss

        if return_dict or (return_dict is None and self.config.use_return_dict):
            return {
                "loss": loss,
                "dollars_pred": dollars_pred,
                "cents_pred": cents_pred,
                "dollars_loss": dollars_loss,
                "cents_loss": cents_loss,
                "hidden_states": encoder_outputs.hidden_states,
                "attentions": encoder_outputs.attentions,
            }
        
        return (loss, dollars_pred, cents_pred) if loss is not None else (dollars_pred, cents_pred)


class MonetaryAmountTrainer(Trainer):
    """
    Custom trainer for monetary amount prediction.
    Handles the regression outputs properly.
    """
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute the loss for our regression model.
        
        Args:
            model: The model to compute loss for
            inputs: The inputs to pass to the model
            return_outputs: Whether to return the outputs along with the loss
            num_items_in_batch: Number of items in the batch (for gradient scaling)
            
        Returns:
            loss or (loss, outputs) if return_outputs is True
        """
        outputs = model(**inputs)
        
        # Loss should be calculated in the model's forward pass
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        
        # Scale loss if needed
        if num_items_in_batch is not None and num_items_in_batch > 0:
            loss = loss * (len(inputs["input_ids"]) / num_items_in_batch)
        
        return (loss, outputs) if return_outputs else loss
    
    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only: bool,
        ignore_keys=None,
    ):
        """
        Perform a prediction step for our regression model.
        Overridden to handle our custom regression outputs.
        """
        has_labels = all(inputs.get(k) is not None for k in self.label_names)

        # Prepare inputs
        inputs = self._prepare_inputs(inputs)
        
        # Perform inference
        with torch.no_grad():
            # Forward pass
            outputs = model(**inputs)
            
            # Get loss if we have labels
            if has_labels:
                if isinstance(outputs, dict):
                    loss = outputs["loss"].mean().detach()
                else:
                    loss = outputs[0].mean().detach()
            else:
                loss = None
            
            # Get predictions
            if isinstance(outputs, dict):
                dollars_pred = outputs["dollars_pred"]
                cents_pred = outputs["cents_pred"]
            else:
                dollars_pred = outputs[1]
                cents_pred = outputs[2]
            
            # Stack predictions for compatibility with trainer
            preds = torch.stack([dollars_pred, cents_pred], dim=1)

        # Return loss, predictions, and labels
        if prediction_loss_only:
            return (loss, None, None)

        if has_labels:
            labels = inputs["labels"]
        else:
            labels = None

        return (loss, preds, labels)

    def evaluation_loop(
        self,
        dataloader,
        description,
        prediction_loss_only=None,
        ignore_keys=None,
        metric_key_prefix="eval",
    ):
        """
        Overridden evaluation loop to handle our regression outputs.
        """
        # Initialize metrics
        args = self.args
        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only
        
        model = self._wrap_model(self.model, training=False, dataloader=dataloader)
        
        batch_size = dataloader.batch_size
        eval_losses = []
        preds_host = None
        labels_host = None
        
        model.eval()
        
        for step, inputs in enumerate(dataloader):
            # Prediction step
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            
            # Update metrics
            if loss is not None:
                eval_losses.append(loss.item())
            
            # Gather predictions and labels
            if logits is not None:
                preds_host = logits if preds_host is None else torch.cat((preds_host, logits), dim=0)
            if labels is not None:
                labels_host = labels if labels_host is None else torch.cat((labels_host, labels), dim=0)
        
        # Compute metrics
        metrics = {}
        if loss is not None:
            metrics[f"{metric_key_prefix}_loss"] = torch.tensor(eval_losses).mean().item()
        
        if preds_host is not None and labels_host is not None:
            # Ensure tensors are on CPU and float32
            preds_host = preds_host.cpu().to(torch.float32)
            labels_host = labels_host.cpu().to(torch.float32)
            
            # Compute MSE for dollars and cents separately
            dollars_mse = nn.functional.mse_loss(preds_host[:, 0], labels_host[:, 0]).item()
            cents_mse = nn.functional.mse_loss(preds_host[:, 1], labels_host[:, 1]).item()
            
            metrics.update({
                f"{metric_key_prefix}_dollars_mse": dollars_mse,
                f"{metric_key_prefix}_cents_mse": cents_mse,
                f"{metric_key_prefix}_combined_mse": (dollars_mse + cents_mse) / 2,
            })
        
        # Return in the format expected by Trainer
        return trainer_utils.EvalLoopOutput(
            predictions=preds_host,
            label_ids=labels_host,
            metrics=metrics,
            num_samples=preds_host.shape[0] if preds_host is not None else 0
        )


def initialize_model(model_name: Optional[str] = None, device: Optional[Union[str, torch.device]] = None) -> MonetaryAmountModel:
    """
    Initialize the monetary amount prediction model.

    Args:
        model_name (str, optional): Name of the pretrained T5 model to use as encoder. Defaults to config.MODEL_NAME.
        device (str or torch.device, optional): Device to place the model on. Defaults to config.DEVICE.

    Returns:
        MonetaryAmountModel: Initialized model
    """
    if model_name is None:
        model_name = config.MODEL_NAME

    if device is None:
        device = config.DEVICE

    logger.info(f"Initializing monetary amount model with {model_name} encoder on {device}")

    # Load the T5 configuration
    t5_config = T5Config.from_pretrained(model_name)
    
    # Set use_cache to False to avoid generation-related issues
    t5_config.use_cache = False

    # Initialize our custom model
    model = MonetaryAmountModel(t5_config)

    # Load the pretrained encoder weights
    encoder_model = T5EncoderModel.from_pretrained(model_name)
    model.encoder.load_state_dict(encoder_model.state_dict())

    # Move the model to the device
    model = model.to(device)

    # Set model to evaluation mode by default
    model.eval()

    return model


def save_model(
    model: MonetaryAmountModel,
    tokenizer: AutoTokenizer,
    output_dir: str,
    metadata: Optional[Dict] = None,
) -> str:
    """
    Save the monetary amount model and tokenizer with configuration details.

    Args:
        model (MonetaryAmountModel): Model to save
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
            "model_type": "monetary_amount_model",  # Custom type
            "vocab_size": model.config.vocab_size,
            "hidden_size": model.config.hidden_size,
            "num_layers": model.encoder.config.num_layers,
            "device": str(next(model.parameters()).device),
            "max_length": config.MAX_INPUT_LENGTH,
        }
    )

    # Save metadata
    import json

    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    return output_dir


def load_model(model_path: str, device: Optional[Union[str, torch.device]] = None) -> tuple:
    """
    Load the monetary amount model and tokenizer from a saved checkpoint.

    Args:
        model_path (str): Path to the saved model
        device (str or torch.device, optional): Device to place the model on. Defaults to config.DEVICE.

    Returns:
        tuple: (model, tokenizer, metadata)
    """
    if device is None:
        device = config.DEVICE

    logger.info(f"Loading model from {model_path} to {device}")

    # Load the model
    model = MonetaryAmountModel.from_pretrained(model_path)
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
    model: MonetaryAmountModel,
    tokenizer: AutoTokenizer,
    texts: List[str],
    max_length: Optional[int] = None,
) -> List[Dict[str, float]]:
    """
    Process a batch of input texts for monetary amount prediction.

    Args:
        model (MonetaryAmountModel): The monetary amount prediction model
        tokenizer (AutoTokenizer): Tokenizer for the model
        texts (List[str]): List of input texts to process
        max_length (int, optional): Maximum input length. Defaults to config.MAX_INPUT_LENGTH.

    Returns:
        List[Dict[str, float]]: List of predictions, each containing 'dollars' and 'cents' values
    """
    if max_length is None:
        max_length = config.MAX_INPUT_LENGTH

    # Prepare batch inputs
    batch_inputs = prepare_inputs(tokenizer, texts, max_length)

    # Move inputs to the same device as the model
    device = next(model.parameters()).device
    batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}

    # Set model to evaluation mode
    model.eval()

    # Process batch
    with torch.no_grad():
        dollars_pred, cents_pred = model(**batch_inputs)

    # Convert predictions to list of dictionaries
    predictions = []
    for d, c in zip(dollars_pred.cpu().numpy(), cents_pred.cpu().numpy()):
        predictions.append({
            "dollars": float(d),
            "cents": float(c)
        })

    return predictions


def generate_text(
    model: MonetaryAmountModel,
    tokenizer: AutoTokenizer,
    input_text: str,
    max_length: Optional[int] = None,
) -> Dict[str, float]:
    """
    Generate monetary amount prediction for a single input text.

    Args:
        model (MonetaryAmountModel): The monetary amount prediction model
        tokenizer (AutoTokenizer): Tokenizer for the model
        input_text (str): Input text to process
        max_length (int, optional): Maximum input length. Defaults to config.MAX_INPUT_LENGTH.

    Returns:
        Dict[str, float]: Prediction containing 'dollars' and 'cents' values
    """
    # Process single text using batch_process
    predictions = batch_process(model, tokenizer, [input_text], max_length)
    return predictions[0]


def test_model():
    """Test the model with a sample input."""
    # Initialize model and tokenizer
    model = initialize_model()
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

    # Sample input
    input_text = "twenty-five dollars and ten cents"

    # Generate prediction
    prediction = generate_text(model, tokenizer, input_text)

    logger.info(f"Input: {input_text}")
    logger.info(f"Predicted amount: ${prediction['dollars']:.0f}.{prediction['cents']:.0f}")

    return prediction


class MonetaryAmountDataCollator:
    """
    Data collator for monetary amount prediction.
    Converts the targets into the format expected by the model.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        # First, separate the inputs and targets
        input_texts = [f["input"] for f in features]
        
        # Create a batch of 2 values per example: [dollars, cents]
        labels = torch.tensor(
            [[f["target"]["dollars"], f["target"]["cents"]] for f in features],
            dtype=torch.float32  # Explicitly use float32
        )

        # Tokenize all the texts together
        batch = self.tokenizer(
            input_texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        # Add the targets
        batch["labels"] = labels

        return batch


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Test the model configuration
    test_model()
