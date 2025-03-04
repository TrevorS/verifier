#!/usr/bin/env python
"""
Training script for the character-level transformer model.
"""

import argparse
import logging
import os
import time
from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils.data import random_split, Subset

import config
from char_model import (
    CharacterEncoder,
    CharacterTransformerModel,
    CharacterMonetaryAmountDataset,
    CharacterMonetaryAmountTrainer,
    generate_prediction
)

# Configure logger
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a character-level transformer model for monetary amount prediction")
    
    # Data arguments
    parser.add_argument("--train_data", type=str, default=str(config.TRAIN_DATA_PATH),
                        help="Path to training data file")
    parser.add_argument("--val_data", type=str, default=str(config.VAL_DATA_PATH),
                        help="Path to validation data file")
    parser.add_argument("--test_data", type=str, default=str(config.TEST_DATA_PATH),
                        help="Path to test data file")
    parser.add_argument("--subset_size", type=int, default=None,
                        help="Use a subset of the data for faster experimentation")
    parser.add_argument("--normalize_targets", action="store_true", default=True,
                        help="Normalize target values")
    parser.add_argument("--log_transform", action="store_true", default=True,
                        help="Apply log(x+1) transformation to dollar values")
    
    # Model arguments
    parser.add_argument("--embedding_dim", type=int, default=128,
                        help="Dimension of character embeddings")
    parser.add_argument("--hidden_dim", type=int, default=256,
                        help="Dimension of hidden layers")
    parser.add_argument("--num_heads", type=int, default=4,
                        help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=3,
                        help="Number of transformer layers")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate")
    parser.add_argument("--max_length", type=int, default=200,
                        help="Maximum sequence length")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay for regularization")
    parser.add_argument("--num_epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--save_dir", type=str, default=str(config.MODELS_DIR / "char_model"),
                        help="Directory to save model checkpoints")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Number of workers for data loading (0 for main process)")
    
    # Other arguments
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default=config.DEVICE,
                        help="Device to train on (cuda, mps, or cpu)")
    parser.add_argument("--test_only", action="store_true",
                        help="Only run testing on the test set")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to a saved model to load for testing")
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="Validation split ratio if no validation set is provided")
    parser.add_argument("--force_cpu", action="store_true",
                        help="Force using CPU even if GPU is available")
    parser.add_argument("--verbose", action="store_true",
                        help="Show progress bars and detailed logging")
    
    return parser.parse_args()


class CustomCharacterMonetaryAmountTrainer:
    """Custom trainer for the character-level transformer model."""
    
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
        num_workers=0,
        verbose=False,
    ):
        """Initialize the trainer."""
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.save_dir = save_dir
        self.num_workers = num_workers
        self.verbose = verbose
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Create learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=2,
            verbose=True
        )
        
        # Initialize data loaders
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True if self.device != "cpu" else False
        )
        
        if self.val_dataset is not None:
            self.val_loader = torch.utils.data.DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True if self.device != "cpu" else False
            )
        else:
            self.val_loader = None
        
        # Initialize metrics
        self.metrics = {
            "train_loss": [],
            "val_loss": [],
            "val_dollars_mse": [],
            "val_cents_mse": []
        }
        
        # Initialize best model state
        self.best_val_loss = float("inf")
        self.best_model_state = None
        
    def train(self):
        """Train the model."""
        # Create save directory if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)
        
        for epoch in range(self.num_epochs):
            # Training loop
            self.model.train()
            train_loss = 0.0
            
            # Create progress bar for training
            start_time = time.time()
            train_iter = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}", disable=not self.verbose)
            
            for batch in train_iter:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs["loss"]
                
                # Backward pass
                loss.backward()
                
                # Update parameters
                self.optimizer.step()
                
                # Accumulate loss
                train_loss += loss.item()
                
                if self.verbose:
                    train_iter.set_postfix({"loss": loss.item()})
            
            # Calculate average training loss
            avg_train_loss = train_loss / len(self.train_loader)
            self.metrics["train_loss"].append(avg_train_loss)
            
            # Validation loop
            if self.val_loader:
                val_metrics = self.evaluate()
                val_loss = val_metrics["val_loss"]
                self.metrics["val_loss"].append(val_loss)
                self.metrics["val_dollars_mse"].append(val_metrics["val_dollars_mse"])
                self.metrics["val_cents_mse"].append(val_metrics["val_cents_mse"])
                
                # Update learning rate scheduler
                if hasattr(self, 'scheduler'):
                    self.scheduler.step(val_loss)
                
                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_model_state = self.model.state_dict()
                    self.save_model(os.path.join(self.save_dir, "best_model.pt"))
                    
                # Calculate epoch time
                epoch_time = time.time() - start_time
                
                # Log epoch metrics
                logger.info(f"Epoch {epoch+1}/{self.num_epochs} - "
                           f"Time: {epoch_time:.2f}s - "
                           f"Train Loss: {avg_train_loss:.4f} - "
                           f"Val Loss: {val_loss:.4f} - "
                           f"Val Dollars MSE: {val_metrics['val_dollars_mse']:.4f} - "
                           f"Val Cents MSE: {val_metrics['val_cents_mse']:.4f}")
            else:
                # Log epoch metrics without validation
                logger.info(f"Epoch {epoch+1}/{self.num_epochs} - "
                           f"Train Loss: {avg_train_loss:.4f}")
        
        # Save final model
        self.save_model(os.path.join(self.save_dir, "final_model.pt"))
        
        # If we have a best model state, restore it
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        return self.metrics
    
    def evaluate(self):
        """Evaluate the model on the validation set."""
        self.model.eval()
        val_loss = 0.0
        dollars_mse = 0.0
        cents_mse = 0.0
        
        # Create progress bar for validation
        val_iter = tqdm(self.val_loader, desc="Validation", disable=not self.verbose)
        
        with torch.no_grad():
            for batch in val_iter:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                
                # Accumulate metrics
                val_loss += outputs["loss"].item()
                dollars_mse += outputs["dollars_loss"].item()
                cents_mse += outputs["cents_loss"].item()
                
                if self.verbose:
                    val_iter.set_postfix({"loss": outputs["loss"].item()})
        
        # Calculate average metrics
        num_batches = len(self.val_loader)
        avg_val_loss = val_loss / num_batches
        avg_dollars_mse = dollars_mse / num_batches
        avg_cents_mse = cents_mse / num_batches
        
        return {
            "val_loss": avg_val_loss,
            "val_dollars_mse": avg_dollars_mse,
            "val_cents_mse": avg_cents_mse
        }
    
    def save_model(self, path):
        """Save the model to disk."""
        # Get model configuration
        model_config = {
            "vocab_size": self.model.vocab_size,
            "embedding_dim": self.model.embedding_dim,
            "hidden_dim": self.model.hidden_dim,
            "num_heads": self.model.num_heads,
            "num_layers": self.model.num_layers,
            "max_length": self.model.max_length
        }
        
        # Save model state and configuration
        save_dict = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "model_config": model_config
        }
        
        # Add scheduler state if it exists
        if hasattr(self, 'scheduler'):
            save_dict["scheduler_state_dict"] = self.scheduler.state_dict()
        
        torch.save(save_dict, path)
        
        logger.info(f"Model saved to {path}")
        
        return path


def train_model(args):
    """Train the character-level transformer model."""
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Check for MPS compatibility issues
    if args.device == "mps" and not args.force_cpu:
        if not os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"):
            logger.warning(
                "MPS device detected but PYTORCH_ENABLE_MPS_FALLBACK environment variable not set. "
                "Some operations may not be supported on MPS. "
                "Consider setting PYTORCH_ENABLE_MPS_FALLBACK=1 or using --force_cpu."
            )
            logger.info("Setting device to CPU to avoid compatibility issues.")
            args.device = "cpu"
    
    # Initialize character encoder
    char_encoder = CharacterEncoder(max_length=args.max_length)
    
    # Load datasets
    logger.info(f"Loading training data from {args.train_data}")
    train_dataset = CharacterMonetaryAmountDataset(
        args.train_data,
        char_encoder=char_encoder,
        max_length=args.max_length,
        normalize_targets=args.normalize_targets,
        log_transform=args.log_transform
    )
    
    # Use a subset of the data if specified
    if args.subset_size is not None:
        subset_size = min(args.subset_size, len(train_dataset))
        logger.info(f"Using a subset of {subset_size} examples for training")
        indices = torch.randperm(len(train_dataset))[:subset_size]
        train_dataset = Subset(train_dataset, indices)
    
    # Load or create validation dataset
    if os.path.exists(args.val_data):
        logger.info(f"Loading validation data from {args.val_data}")
        val_dataset = CharacterMonetaryAmountDataset(
            args.val_data,
            char_encoder=char_encoder,
            max_length=args.max_length,
            normalize_targets=args.normalize_targets,
            log_transform=args.log_transform
        )
        
        # Use a subset of the validation data if specified
        if args.subset_size is not None:
            subset_size = min(args.subset_size // 4, len(val_dataset))
            logger.info(f"Using a subset of {subset_size} examples for validation")
            indices = torch.randperm(len(val_dataset))[:subset_size]
            val_dataset = Subset(val_dataset, indices)
    else:
        logger.info(f"No validation data found at {args.val_data}, creating split from training data")
        # Calculate split sizes
        val_size = int(len(train_dataset) * args.val_split)
        train_size = len(train_dataset) - val_size
        
        # Split the dataset
        train_dataset, val_dataset = random_split(
            train_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(args.seed)
        )
        
        logger.info(f"Split dataset: {train_size} training examples, {val_size} validation examples")
    
    # Initialize model
    logger.info("Initializing character-level transformer model")
    model = CharacterTransformerModel(
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        max_length=args.max_length
    )
    
    # Log model architecture
    logger.info(f"Model architecture: Character-level transformer with {args.num_layers} layers")
    logger.info(f"Embedding dimension: {args.embedding_dim}, Hidden dimension: {args.hidden_dim}")
    logger.info(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    logger.info(f"Using device: {args.device}")
    logger.info(f"Target normalization: {args.normalize_targets}, Log transform: {args.log_transform}")
    
    # Initialize trainer
    trainer = CustomCharacterMonetaryAmountTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epochs,
        device=args.device,
        save_dir=args.save_dir,
        num_workers=args.num_workers,
        verbose=args.verbose
    )
    
    # Train the model
    logger.info("Starting training")
    metrics = trainer.train()
    
    # Log final metrics
    logger.info(f"Training completed. Final metrics:")
    logger.info(f"  Train loss: {metrics['train_loss'][-1]:.4f}")
    logger.info(f"  Validation loss: {metrics['val_loss'][-1]:.4f}")
    logger.info(f"  Validation dollars MSE: {metrics['val_dollars_mse'][-1]:.4f}")
    logger.info(f"  Validation cents MSE: {metrics['val_cents_mse'][-1]:.4f}")
    
    return model, char_encoder, train_dataset


def test_model(args, model=None, char_encoder=None, train_dataset=None):
    """Test the character-level transformer model on the test set."""
    # Load model if not provided
    if model is None:
        if args.model_path:
            model_path = args.model_path
        else:
            model_path = os.path.join(args.save_dir, "best_model.pt")
            
        logger.info(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=args.device)
        
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
        model = model.to(args.device)
        model.eval()
    
    # Initialize character encoder if not provided
    if char_encoder is None:
        char_encoder = CharacterEncoder(max_length=args.max_length)
    
    # Load test dataset
    logger.info(f"Loading test data from {args.test_data}")
    test_dataset = CharacterMonetaryAmountDataset(
        args.test_data,
        char_encoder=char_encoder,
        max_length=args.max_length,
        normalize_targets=args.normalize_targets,
        log_transform=args.log_transform
    )
    
    # Use a subset of the test data if specified
    if args.subset_size is not None:
        subset_size = min(args.subset_size // 4, len(test_dataset))
        logger.info(f"Using a subset of {subset_size} examples for testing")
        indices = torch.randperm(len(test_dataset))[:subset_size]
        test_dataset = Subset(test_dataset, indices)
    
    # Create test data loader
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if args.device != "cpu" else False
    )
    
    # Evaluate on test set
    model.eval()
    test_loss = 0.0
    dollars_mse = 0.0
    cents_mse = 0.0
    
    # Create progress bar for testing
    test_iter = tqdm(test_loader, desc="Testing", disable=not args.verbose)
    
    with torch.no_grad():
        for batch in test_iter:
            # Move batch to device
            batch = {k: v.to(args.device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            
            # Accumulate metrics
            test_loss += outputs["loss"].item()
            dollars_mse += outputs["dollars_loss"].item()
            cents_mse += outputs["cents_loss"].item()
            
            if args.verbose:
                test_iter.set_postfix({"loss": outputs["loss"].item()})
    
    # Calculate average metrics
    num_batches = len(test_loader)
    avg_test_loss = test_loss / num_batches
    avg_dollars_mse = dollars_mse / num_batches
    avg_cents_mse = cents_mse / num_batches
    
    # Log test metrics
    logger.info(f"Test metrics:")
    logger.info(f"  Test loss: {avg_test_loss:.4f}")
    logger.info(f"  Test dollars MSE: {avg_dollars_mse:.4f}")
    logger.info(f"  Test cents MSE: {avg_cents_mse:.4f}")
    
    # Test on some examples
    logger.info("Testing on some examples:")
    test_examples = [
        "twenty-five dollars and ten cents",
        "one hundred dollars and zero cents",
        "five dollars and ninety-nine cents",
        "two thousand five hundred dollars and twenty-five cents",
        "zero dollars and fifty cents"
    ]
    
    # Use the original dataset for denormalization if available
    dataset_for_denorm = train_dataset if train_dataset is not None else test_dataset
    if isinstance(dataset_for_denorm, Subset):
        # If it's a subset, get the original dataset
        dataset_for_denorm = dataset_for_denorm.dataset
    
    for example in test_examples:
        prediction = generate_prediction(model, char_encoder, example, device=args.device, dataset=dataset_for_denorm)
        logger.info(f"  Input: {example}")
        # Format the prediction as $dollars.cents
        dollars = int(prediction['dollars'])  # Convert to integer for dollars
        cents = int(prediction['cents'])      # Convert to integer for cents
        logger.info(f"  Predicted: ${dollars}.{cents:02d}")  # Format cents with leading zero if needed
    
    return {
        "test_loss": avg_test_loss,
        "test_dollars_mse": avg_dollars_mse,
        "test_cents_mse": avg_cents_mse
    }


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(args.save_dir, "training.log"))
        ]
    )
    
    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Log arguments
    logger.info("Training arguments:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")
    
    # Train or test model
    if args.test_only:
        test_model(args)
    else:
        model, char_encoder, train_dataset = train_model(args)
        test_model(args, model, char_encoder, train_dataset)
    
    logger.info("Done!")


if __name__ == "__main__":
    main() 
