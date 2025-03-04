"""
Character-level transformer model for monetary amount prediction.
This model processes input text character by character without using a tokenizer.
"""

import logging
import os
import math
from typing import Dict, List, Optional, Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import config

logger = logging.getLogger(__name__)

# Define the character vocabulary
# We include all ASCII printable characters, plus special tokens
CHAR_VOCAB = {
    # Special tokens
    "<PAD>": 0,  # Padding token
    "<UNK>": 1,  # Unknown character token
    
    # Standard ASCII printable characters (32-126)
    **{chr(i): i-30 for i in range(32, 127)},
}

# Create a reverse mapping for decoding
IDX_TO_CHAR = {idx: char for char, idx in CHAR_VOCAB.items()}

# Maximum sequence length for character inputs
MAX_CHAR_LENGTH = 200  # Adjust based on your data


class CharacterEncoder:
    """
    Encodes text as character-level indices.
    """
    
    def __init__(self, char_vocab=CHAR_VOCAB, max_length=MAX_CHAR_LENGTH):
        self.char_vocab = char_vocab
        self.max_length = max_length
        self.pad_token_id = char_vocab["<PAD>"]
        self.unk_token_id = char_vocab["<UNK>"]
        
    def encode(self, text: str) -> Dict[str, torch.Tensor]:
        """
        Encode a string into character indices.
        
        Args:
            text: Input string to encode
            
        Returns:
            Dictionary with input_ids and attention_mask tensors
        """
        # Convert string to character indices
        char_ids = []
        for char in text[:self.max_length]:
            char_ids.append(self.char_vocab.get(char, self.unk_token_id))
            
        # Create attention mask (1 for actual characters, 0 for padding)
        attention_mask = [1] * len(char_ids)
        
        # Pad sequences to max_length
        padding_length = self.max_length - len(char_ids)
        if padding_length > 0:
            char_ids = char_ids + [self.pad_token_id] * padding_length
            attention_mask = attention_mask + [0] * padding_length
            
        return {
            "input_ids": torch.tensor(char_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long)
        }
    
    def batch_encode(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """
        Encode a batch of strings into character indices.
        
        Args:
            texts: List of input strings to encode
            
        Returns:
            Dictionary with batched input_ids and attention_mask tensors
        """
        batch_input_ids = []
        batch_attention_mask = []
        
        for text in texts:
            encoded = self.encode(text)
            batch_input_ids.append(encoded["input_ids"])
            batch_attention_mask.append(encoded["attention_mask"])
            
        return {
            "input_ids": torch.stack(batch_input_ids),
            "attention_mask": torch.stack(batch_attention_mask)
        }


class PositionalEncoding(nn.Module):
    """
    Positional encoding for the transformer model.
    """
    
    def __init__(self, d_model, max_len=MAX_CHAR_LENGTH):
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register buffer to be saved with model
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        """
        Add positional encoding to input embeddings.
        
        Args:
            x: Input embeddings [batch_size, seq_len, embedding_dim]
            
        Returns:
            Embeddings with positional encoding added
        """
        return x + self.pe[:, :x.size(1)]


class CharacterTransformerModel(nn.Module):
    """
    Character-level transformer model for monetary amount prediction.
    Processes input text character by character and outputs dollar and cent values.
    """
    
    def __init__(
        self,
        vocab_size=len(CHAR_VOCAB),
        embedding_dim=128,
        hidden_dim=256,
        num_heads=4,
        num_layers=3,
        dropout=0.1,
        max_length=MAX_CHAR_LENGTH
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_length = max_length
        
        # Character embedding layer
        self.char_embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(embedding_dim, max_length)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Sequence pooling layer (attention-based)
        self.attention_pooling = nn.Sequential(
            nn.Linear(embedding_dim, 1),
            nn.Softmax(dim=1)
        )
        
        # Regression heads for dollars and cents
        self.dollars_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        self.cents_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
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
            input_ids: Tensor of character indices [batch_size, seq_len]
            attention_mask: Tensor of attention mask [batch_size, seq_len]
            labels: Optional tensor of shape (batch_size, 2) containing [dollars, cents] targets
            dollars_target: Optional tensor of target dollar values (for direct use)
            cents_target: Optional tensor of target cent values (for direct use)
            return_dict: Whether to return a dictionary
            
        Returns:
            During training (if targets provided):
                Dict containing loss and predictions
            During inference:
                Tuple of (dollars_pred, cents_pred) if return_dict is False
                Dict containing predictions if return_dict is True
        """
        # Get character embeddings
        embeddings = self.char_embedding(input_ids)  # [batch_size, seq_len, embedding_dim]
        
        # Add positional encoding
        embeddings = self.positional_encoding(embeddings)
        
        # Create attention mask for transformer (1 = don't mask, 0 = mask)
        # Convert from [batch_size, seq_len] to [batch_size, seq_len, seq_len]
        if attention_mask is not None:
            # Create a mask for padded positions (1 = don't mask, 0 = mask)
            transformer_mask = attention_mask.bool()
        else:
            transformer_mask = None
        
        # Pass through transformer encoder
        # Handle MPS compatibility issues
        device = embeddings.device
        if device.type == "mps" and not os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"):
            # Fall back to CPU for transformer encoder
            cpu_embeddings = embeddings.to("cpu")
            cpu_transformer_mask = None if transformer_mask is None else transformer_mask.to("cpu")
            cpu_sequence_output = self.transformer_encoder(
                cpu_embeddings, 
                src_key_padding_mask=None if cpu_transformer_mask is None else ~cpu_transformer_mask
            )
            sequence_output = cpu_sequence_output.to(device)
        else:
            # Use the original device
            sequence_output = self.transformer_encoder(
                embeddings, 
                src_key_padding_mask=None if transformer_mask is None else ~transformer_mask
            )
        
        # Apply attention pooling to get a weighted sum of all character representations
        # First compute attention weights
        attention_weights = self.attention_pooling(sequence_output)  # [batch_size, seq_len, 1]
        
        # Apply attention mask to zero out padding tokens
        if attention_mask is not None:
            # Expand attention mask to match attention_weights shape
            expanded_attention_mask = attention_mask.unsqueeze(-1)  # [batch_size, seq_len, 1]
            attention_weights = attention_weights * expanded_attention_mask
            # Re-normalize the weights
            attention_weights = attention_weights / (attention_weights.sum(dim=1, keepdim=True) + 1e-9)
        
        # Apply attention weights to get the context vector
        context_vector = torch.sum(sequence_output * attention_weights, dim=1)  # [batch_size, embedding_dim]
        
        # Get predictions from both heads
        dollars_pred = self.dollars_head(context_vector)
        cents_pred = self.cents_head(context_vector)
        
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

        if return_dict or return_dict is None:
            return {
                "loss": loss,
                "dollars_pred": dollars_pred,
                "cents_pred": cents_pred,
                "dollars_loss": dollars_loss,
                "cents_loss": cents_loss,
            }
        
        return (loss, dollars_pred, cents_pred) if loss is not None else (dollars_pred, cents_pred)


class CharacterMonetaryAmountDataset(Dataset):
    """
    Dataset for character-level monetary amount prediction.
    """
    
    def __init__(self, data_path, char_encoder=None, max_length=MAX_CHAR_LENGTH, normalize_targets=True, log_transform=True):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to the data file (jsonl format)
            char_encoder: Character encoder to use
            max_length: Maximum sequence length
            normalize_targets: Whether to normalize target values
            log_transform: Whether to apply log(x+1) transformation to dollar values
        """
        self.data_path = data_path
        self.max_length = max_length
        self.char_encoder = char_encoder or CharacterEncoder(max_length=max_length)
        self.normalize_targets = normalize_targets
        self.log_transform = log_transform
        
        # Load data
        self.data = self._load_data()
        
        # Calculate statistics for normalization
        if self.normalize_targets:
            self._calculate_normalization_stats()
        
    def _calculate_normalization_stats(self):
        """
        Calculate statistics for normalizing target values.
        """
        dollars = [item['target']['dollars'] for item in self.data]
        cents = [item['target']['cents'] for item in self.data]
        
        # Apply log transform to dollars if enabled
        if self.log_transform:
            dollars = [math.log(d + 1.0) for d in dollars]
            logger.info(f"Applied log(x+1) transformation to dollar values")
        
        # Calculate mean and std for dollars
        self.dollars_mean = sum(dollars) / len(dollars)
        self.dollars_std = (sum((x - self.dollars_mean) ** 2 for x in dollars) / len(dollars)) ** 0.5
        self.dollars_std = max(self.dollars_std, 1.0)  # Avoid division by zero
        
        # Calculate mean and std for cents
        self.cents_mean = sum(cents) / len(cents)
        self.cents_std = (sum((x - self.cents_mean) ** 2 for x in cents) / len(cents)) ** 0.5
        self.cents_std = max(self.cents_std, 1.0)  # Avoid division by zero
        
        logger.info(f"Target normalization stats:")
        if self.log_transform:
            logger.info(f"  Log-Dollars: mean={self.dollars_mean:.4f}, std={self.dollars_std:.4f}")
        else:
            logger.info(f"  Dollars: mean={self.dollars_mean:.2f}, std={self.dollars_std:.2f}")
        logger.info(f"  Cents: mean={self.cents_mean:.2f}, std={self.cents_std:.2f}")
    
    def normalize_dollar(self, value):
        """Normalize dollar value."""
        if self.log_transform:
            # Apply log transform first
            value = math.log(value + 1.0)
            
        if self.normalize_targets:
            return (value - self.dollars_mean) / self.dollars_std
        
        return value
    
    def denormalize_dollar(self, value):
        """Denormalize dollar value."""
        if self.normalize_targets:
            value = value * self.dollars_std + self.dollars_mean
            
        if self.log_transform:
            # Undo log transform
            value = math.exp(value) - 1.0
            # Ensure non-negative
            value = max(0.0, value)
            
        return value
    
    def normalize_cent(self, value):
        """Normalize cent value."""
        if self.normalize_targets:
            return (value - self.cents_mean) / self.cents_std
        return value
    
    def denormalize_cent(self, value):
        """Denormalize cent value."""
        if self.normalize_targets:
            value = value * self.cents_std + self.cents_mean
        # Ensure cents are in valid range
        value = max(0.0, min(99.0, value))
        return value
        
    def _load_data(self):
        """
        Load data from jsonl file.
        Each line should be a JSON object with 'input' and 'target' fields.
        Target should have 'dollars' and 'cents' fields.
        """
        import json
        
        data = []
        with open(self.data_path, 'r') as f:
            for line in f:
                item = json.loads(line.strip())
                data.append(item)
                
        logger.info(f"Loaded {len(data)} examples from {self.data_path}")
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Get a single example from the dataset.
        
        Returns:
            Dictionary with input_ids, attention_mask, and labels
        """
        item = self.data[idx]
        
        # Encode the input text
        encoded_input = self.char_encoder.encode(item['input'])
        
        # Get the target values and normalize if needed
        dollars = self.normalize_dollar(item['target']['dollars'])
        cents = self.normalize_cent(item['target']['cents'])
        
        target = torch.tensor([dollars, cents], dtype=torch.float32)
        
        return {
            "input_ids": encoded_input["input_ids"],
            "attention_mask": encoded_input["attention_mask"],
            "labels": target
        }


class CharacterMonetaryAmountTrainer:
    """
    Trainer for the character-level monetary amount model.
    """
    
    def __init__(
        self,
        model,
        train_dataset,
        val_dataset=None,
        batch_size=16,
        learning_rate=1e-4,
        weight_decay=0.01,
        num_epochs=10,
        device=None,
        save_dir=None,
    ):
        """
        Initialize the trainer.
        
        Args:
            model: The model to train
            train_dataset: Training dataset
            val_dataset: Validation dataset
            batch_size: Batch size for training
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization
            num_epochs: Number of training epochs
            device: Device to train on
            save_dir: Directory to save model checkpoints
        """
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.device = device or torch.device(config.DEVICE)
        self.save_dir = save_dir or config.MODELS_DIR / "char_model"
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=2,
                pin_memory=True
            )
        else:
            self.val_loader = None
            
        # Create optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Create learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=2,
            verbose=True
        )
        
        # Move model to device
        self.model = self.model.to(self.device)
        
    def train(self):
        """
        Train the model.
        
        Returns:
            Dictionary with training metrics
        """
        # Create save directory if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)
        
        best_val_loss = float('inf')
        metrics = {
            "train_loss": [],
            "val_loss": [],
            "val_dollars_mse": [],
            "val_cents_mse": [],
        }
        
        for epoch in range(self.num_epochs):
            # Training loop
            self.model.train()
            train_loss = 0.0
            
            for batch in self.train_loader:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(**batch)
                loss = outputs["loss"]
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                train_loss += loss.item()
                
            # Calculate average training loss
            avg_train_loss = train_loss / len(self.train_loader)
            metrics["train_loss"].append(avg_train_loss)
            
            # Validation loop
            if self.val_loader:
                val_metrics = self.evaluate()
                val_loss = val_metrics["val_loss"]
                metrics["val_loss"].append(val_loss)
                metrics["val_dollars_mse"].append(val_metrics["val_dollars_mse"])
                metrics["val_cents_mse"].append(val_metrics["val_cents_mse"])
                
                # Update learning rate scheduler
                self.scheduler.step(val_loss)
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_model(os.path.join(self.save_dir, "best_model.pt"))
                    
                logger.info(
                    f"Epoch {epoch+1}/{self.num_epochs} - "
                    f"Train Loss: {avg_train_loss:.4f}, "
                    f"Val Loss: {val_loss:.4f}, "
                    f"Val Dollars MSE: {val_metrics['val_dollars_mse']:.4f}, "
                    f"Val Cents MSE: {val_metrics['val_cents_mse']:.4f}"
                )
            else:
                logger.info(
                    f"Epoch {epoch+1}/{self.num_epochs} - "
                    f"Train Loss: {avg_train_loss:.4f}"
                )
                
            # Save checkpoint
            if (epoch + 1) % 5 == 0 or epoch == self.num_epochs - 1:
                self.save_model(os.path.join(self.save_dir, f"checkpoint_epoch_{epoch+1}.pt"))
                
        # Save final model
        self.save_model(os.path.join(self.save_dir, "final_model.pt"))
        
        return metrics
    
    def evaluate(self):
        """
        Evaluate the model on the validation set.
        
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        val_loss = 0.0
        dollars_mse = 0.0
        cents_mse = 0.0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                
                # Accumulate metrics
                val_loss += outputs["loss"].item()
                dollars_mse += outputs["dollars_loss"].item()
                cents_mse += outputs["cents_loss"].item()
                
        # Calculate average metrics
        num_batches = len(self.val_loader)
        return {
            "val_loss": val_loss / num_batches,
            "val_dollars_mse": dollars_mse / num_batches,
            "val_cents_mse": cents_mse / num_batches,
        }
    
    def save_model(self, path):
        """
        Save the model checkpoint.
        
        Args:
            path: Path to save the model
        """
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "model_config": {
                "vocab_size": self.model.vocab_size,
                "embedding_dim": self.model.embedding_dim,
                "hidden_dim": self.model.hidden_dim,
                "num_heads": self.model.num_heads,
                "num_layers": self.model.num_layers,
                "max_length": self.model.max_length,
            }
        }, path)
        
        logger.info(f"Model saved to {path}")


def initialize_model(
    vocab_size=len(CHAR_VOCAB),
    embedding_dim=128,
    hidden_dim=256,
    num_heads=4,
    num_layers=3,
    dropout=0.1,
    max_length=MAX_CHAR_LENGTH,
    device=None
) -> CharacterTransformerModel:
    """
    Initialize the character-level transformer model.
    
    Args:
        vocab_size: Size of the character vocabulary
        embedding_dim: Dimension of character embeddings
        hidden_dim: Dimension of hidden layers
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        dropout: Dropout rate
        max_length: Maximum sequence length
        device: Device to place the model on
        
    Returns:
        CharacterTransformerModel: Initialized model
    """
    if device is None:
        device = config.DEVICE
        
    logger.info(f"Initializing character-level transformer model on {device}")
    
    # Initialize model
    model = CharacterTransformerModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        max_length=max_length
    )
    
    # Log model architecture
    logger.info(f"Model architecture: Character-level transformer with {num_layers} layers")
    logger.info(f"Embedding dimension: {embedding_dim}, Hidden dimension: {hidden_dim}")
    logger.info(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Move model to device
    model = model.to(device)
    
    # Set model to evaluation mode by default
    model.eval()
    
    return model


def load_model(model_path, device=None):
    """
    Load a saved character-level transformer model.
    
    Args:
        model_path: Path to the saved model
        device: Device to load the model on
        
    Returns:
        CharacterTransformerModel: Loaded model
    """
    if device is None:
        device = config.DEVICE
        
    logger.info(f"Loading model from {model_path} to {device}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get model configuration
    model_config = checkpoint["model_config"]
    
    # Initialize model with saved configuration
    model = CharacterTransformerModel(
        vocab_size=model_config["vocab_size"],
        embedding_dim=model_config["embedding_dim"],
        hidden_dim=model_config["hidden_dim"],
        num_heads=model_config["num_heads"],
        num_layers=model_config["num_layers"],
        max_length=model_config["max_length"]
    )
    
    # Load model state
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # Move model to device
    model = model.to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    logger.info(f"Model loaded successfully. Device: {next(model.parameters()).device}")
    
    return model


def prepare_inputs(
    char_encoder: CharacterEncoder,
    text: Union[str, List[str]],
) -> Dict[str, torch.Tensor]:
    """
    Prepare inputs for inference by encoding characters.
    
    Args:
        char_encoder: Character encoder
        text: Input text or batch of texts
        
    Returns:
        Dict[str, torch.Tensor]: Encoded inputs ready for the model
    """
    if isinstance(text, str):
        return char_encoder.encode(text)
    else:
        return char_encoder.batch_encode(text)


def generate_prediction(
    model: CharacterTransformerModel,
    char_encoder: CharacterEncoder,
    input_text: str,
    device=None,
    dataset=None
) -> Dict[str, float]:
    """
    Generate monetary amount prediction for a single input text.
    
    Args:
        model: The character-level transformer model
        char_encoder: Character encoder
        input_text: Input text to process
        device: Device to run inference on
        dataset: Dataset used for training (for denormalization)
        
    Returns:
        Dict[str, float]: Prediction containing 'dollars' and 'cents' values
    """
    if device is None:
        device = next(model.parameters()).device
        
    # Prepare input
    inputs = prepare_inputs(char_encoder, input_text)
    
    # Move inputs to device
    inputs = {k: v.unsqueeze(0).to(device) for k, v in inputs.items()}
    
    # Set model to evaluation mode
    model.eval()
    
    # Generate prediction
    with torch.no_grad():
        outputs = model(**inputs)
        
        # Handle both tuple and dictionary return types
        if isinstance(outputs, tuple):
            dollars_pred, cents_pred = outputs
        else:
            dollars_pred = outputs["dollars_pred"]
            cents_pred = outputs["cents_pred"]
    
    # Denormalize predictions if dataset is provided
    dollars = dollars_pred.item()
    cents = cents_pred.item()
    
    if dataset is not None and hasattr(dataset, 'normalize_targets') and dataset.normalize_targets:
        dollars = dataset.denormalize_dollar(dollars)
        cents = dataset.denormalize_cent(cents)
    
    # Return prediction
    return {
        "dollars": dollars,
        "cents": cents
    }


def batch_process(
    model: CharacterTransformerModel,
    char_encoder: CharacterEncoder,
    texts: List[str],
    device=None,
    dataset=None
) -> List[Dict[str, float]]:
    """
    Process a batch of input texts for monetary amount prediction.
    
    Args:
        model: The character-level transformer model
        char_encoder: Character encoder
        texts: List of input texts to process
        device: Device to run inference on
        dataset: Dataset used for training (for denormalization)
        
    Returns:
        List[Dict[str, float]]: List of predictions, each containing 'dollars' and 'cents' values
    """
    if device is None:
        device = next(model.parameters()).device
        
    # Prepare batch inputs
    batch_inputs = prepare_inputs(char_encoder, texts)
    
    # Move inputs to device
    batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
    
    # Set model to evaluation mode
    model.eval()
    
    # Process batch
    with torch.no_grad():
        outputs = model(**batch_inputs)
        
        # Handle both tuple and dictionary return types
        if isinstance(outputs, tuple):
            dollars_pred, cents_pred = outputs
        else:
            dollars_pred = outputs["dollars_pred"]
            cents_pred = outputs["cents_pred"]
    
    # Convert predictions to list of dictionaries
    predictions = []
    for d, c in zip(dollars_pred.cpu().numpy(), cents_pred.cpu().numpy()):
        # Denormalize predictions if dataset is provided
        dollars = float(d)
        cents = float(c)
        
        if dataset is not None and hasattr(dataset, 'normalize_targets') and dataset.normalize_targets:
            dollars = dataset.denormalize_dollar(dollars)
            cents = dataset.denormalize_cent(cents)
            
        predictions.append({
            "dollars": dollars,
            "cents": cents
        })
        
    return predictions


def test_model():
    """Test the model with a sample input."""
    # Initialize model and character encoder
    model = initialize_model()
    char_encoder = CharacterEncoder()
    
    # Sample input
    input_text = "twenty-five dollars and ten cents"
    
    # Generate prediction
    prediction = generate_prediction(model, char_encoder, input_text)
    
    logger.info(f"Input: {input_text}")
    logger.info(f"Predicted amount: ${prediction['dollars']:.0f}.{prediction['cents']:.0f}")
    
    return prediction


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    # Test the model
    test_model() 
