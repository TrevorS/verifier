"""
Training pipeline using the HuggingFace Trainer API.
"""

import json
import logging
import os

import evaluate
import numpy as np
import torch
import wandb
from Levenshtein import distance as levenshtein_distance
from transformers import (
    AutoTokenizer,
    EarlyStoppingCallback,
    TrainingArguments,
)

import config
from src.dataset import prepare_dataset
from src.inference import extract_amount
from src.model import initialize_model, save_model, MonetaryAmountTrainer

# Set environment variable to avoid tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configure logger
logger = logging.getLogger(__name__)


def configure_training_args(
    output_dir,
    num_train_epochs=None,
    batch_size=None,
    learning_rate=None,
    weight_decay=None,
    warmup_ratio=None,
    logging_steps=None,
    evaluation_strategy=None,
    eval_steps=None,
    save_steps=None,
    gradient_accumulation_steps=None,
    fp16=None,
    early_stopping_patience=3,
):
    """
    Configure training arguments for the MonetaryAmountTrainer.

    Args:
        output_dir (str or Path): Directory to save model checkpoints
        num_train_epochs (int): Number of training epochs
        batch_size (int): Batch size for training and evaluation
        learning_rate (float): Learning rate
        weight_decay (float): Weight decay
        warmup_ratio (float): Ratio of steps for learning rate warmup
        logging_steps (int): Number of steps between logging updates
        evaluation_strategy (str): Evaluation strategy
        eval_steps (int): Number of steps between evaluations
        save_steps (int): Number of steps between saving checkpoints
        gradient_accumulation_steps (int): Number of steps to accumulate gradients
        fp16 (bool): Whether to use mixed precision training
        early_stopping_patience (int): Number of evaluations with no improvement after which to stop

    Returns:
        transformers.TrainingArguments: Training arguments
    """
    # Use default values from config if not provided
    if num_train_epochs is None:
        num_train_epochs = config.NUM_EPOCHS

    if batch_size is None:
        batch_size = config.BATCH_SIZE

    if learning_rate is None:
        learning_rate = config.LEARNING_RATE

    if weight_decay is None:
        weight_decay = config.WEIGHT_DECAY

    if warmup_ratio is None:
        warmup_ratio = config.WARMUP_RATIO

    if logging_steps is None:
        logging_steps = config.LOGGING_STEPS

    if evaluation_strategy is None:
        # Use the new parameter name if available, otherwise fallback
        evaluation_strategy = getattr(config, "EVAL_STRATEGY", getattr(config, "EVALUATION_STRATEGY", "steps"))

    if eval_steps is None:
        eval_steps = config.EVAL_STEPS

    if save_steps is None:
        save_steps = config.SAVE_STEPS

    if gradient_accumulation_steps is None:
        gradient_accumulation_steps = config.GRADIENT_ACCUMULATION_STEPS

    if fp16 is None:
        fp16 = config.FP16

    # Configure training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        logging_steps=logging_steps,
        evaluation_strategy=evaluation_strategy,
        eval_steps=eval_steps,
        save_steps=save_steps,
        save_total_limit=2,
        gradient_accumulation_steps=gradient_accumulation_steps,
        load_best_model_at_end=True,
        metric_for_best_model="combined_mse",  # Use our custom MSE metric
        greater_is_better=False,  # Lower MSE is better
        fp16=fp16,
        report_to="wandb" if wandb.run is not None else "none",
        push_to_hub=False,
        run_name=f"finetuned-{os.path.basename(output_dir)}-{wandb.util.generate_id()}" if wandb.run is not None else None,
    )

    logger.info(f"Training configuration: {num_train_epochs} epochs, batch size {batch_size}, eval steps {eval_steps}")

    return training_args


def compute_metrics(eval_preds):
    """
    Compute evaluation metrics for the regression model.

    Args:
        eval_preds (tuple): Tuple of (predictions, labels) where:
            - predictions is a tensor of shape (batch_size, 2) containing [dollars, cents] predictions
            - labels is a tensor of shape (batch_size, 2) containing [dollars, cents] targets

    Returns:
        dict: Dictionary of metrics including:
            - dollars_mse: Mean squared error for dollar predictions
            - cents_mse: Mean squared error for cent predictions
            - combined_mse: Combined MSE (average of dollars and cents MSE)
    """
    predictions, labels = eval_preds
    
    # Ensure predictions and labels are on CPU
    if hasattr(predictions, "device") and str(predictions.device) != "cpu":
        predictions = predictions.cpu()
    if hasattr(labels, "device") and str(labels.device) != "cpu":
        labels = labels.cpu()
    
    # Convert to numpy if needed
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()
    
    # Calculate MSE for dollars and cents separately
    dollars_mse = np.mean((predictions[:, 0] - labels[:, 0]) ** 2)
    cents_mse = np.mean((predictions[:, 1] - labels[:, 1]) ** 2)
    
    # Calculate combined MSE
    combined_mse = (dollars_mse + cents_mse) / 2
    
    # Prepare metrics
    metrics = {
        "dollars_mse": dollars_mse,
        "cents_mse": cents_mse,
        "combined_mse": combined_mse,
    }
    
    # Log metrics to Weights & Biases if enabled
    if wandb.run is not None:
        wandb.log(metrics)
    
    return metrics


def setup_trainer(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    data_collator,
    output_dir,
    early_stopping_patience=3,
):
    """
    Set up the MonetaryAmountTrainer for fine-tuning.

    Args:
        model (transformers.PreTrainedModel): Model to train
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer
        train_dataset (datasets.Dataset): Training dataset
        eval_dataset (datasets.Dataset): Evaluation dataset
        data_collator (transformers.DataCollator): Data collator
        output_dir (str or Path): Directory to save model checkpoints
        early_stopping_patience (int): Number of evaluations with no improvement after which to stop

    Returns:
        MonetaryAmountTrainer: Configured trainer
    """
    # Configure training arguments
    training_args = configure_training_args(output_dir)

    # Set up trainer
    trainer = MonetaryAmountTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
    )

    # Attach tokenizer to model for use in compute_metrics
    trainer.model.tokenizer = tokenizer

    return trainer


def train_model(
    model_name=None,
    train_data_path=None,
    val_data_path=None,
    output_dir=None,
    wandb_logging=True,
    early_stopping_patience=3,
    quick_test=False,
):
    """
    Train the model on the provided dataset.

    Args:
        model_name (str): Name of the pretrained model
        train_data_path (str or Path): Path to the training data
        val_data_path (str or Path): Path to the validation data
        output_dir (str or Path): Directory to save model checkpoints
        wandb_logging (bool): Whether to log to Weights & Biases
        early_stopping_patience (int): Number of evaluations with no improvement after which to stop
        quick_test (bool): If True, runs a quick test with a small subset of data and reduced steps

    Returns:
        str: Path to the saved model
    """
    # Use default values from config if not provided
    if model_name is None:
        model_name = config.MODEL_NAME

    if train_data_path is None:
        train_data_path = config.TRAIN_DATA_PATH

    if val_data_path is None:
        val_data_path = config.VAL_DATA_PATH

    if output_dir is None:
        output_dir = config.MODELS_DIR / "checkpoints"

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Set quick test parameters if enabled
    training_args_kwargs = {}
    if quick_test:
        logger.info("🧪 Running in quick test mode with reduced dataset and parameters")
        training_args_kwargs = {
            "num_train_epochs": 1,
            "batch_size": 4,
            "logging_steps": 5,
            "evaluation_strategy": "steps",
            "eval_steps": 10,
            "save_steps": 10,
        }

    # Initialize W&B
    if wandb_logging:
        wandb.init(
            project=config.WANDB_PROJECT,
            entity=config.WANDB_ENTITY,
            config={
                "model_name": model_name,
                "learning_rate": config.LEARNING_RATE,
                "batch_size": training_args_kwargs.get("batch_size", config.BATCH_SIZE),
                "epochs": training_args_kwargs.get("num_train_epochs", config.NUM_EPOCHS),
                "warmup_ratio": config.WARMUP_RATIO,
                "weight_decay": config.WEIGHT_DECAY,
                "max_input_length": config.MAX_INPUT_LENGTH,
                "max_target_length": config.MAX_TARGET_LENGTH,
                "early_stopping_patience": early_stopping_patience,
                "quick_test": quick_test,
                "device": config.DEVICE,
            },
        )

    # Prepare dataset
    logger.info("Preparing datasets...")
    preprocessed_dataset, tokenizer, data_collator = prepare_dataset(
        train_data_path,
        val_data_path,
        model_name,
    )

    # Limit dataset size for quick test
    if quick_test:
        # Take a small subset of the data for a quick test
        train_size = min(100, len(preprocessed_dataset["train"]))
        val_size = min(20, len(preprocessed_dataset["validation"]))

        # Create smaller datasets
        preprocessed_dataset["train"] = preprocessed_dataset["train"].select(range(train_size))
        preprocessed_dataset["validation"] = preprocessed_dataset["validation"].select(range(val_size))

        logger.info(f"Using {train_size} training examples and {val_size} validation examples for quick test")

    # Initialize model
    logger.info(f"Initializing model: {model_name}")
    model = initialize_model(model_name)

    # Store tokenizer for compute_metrics function
    compute_metrics.tokenizer = tokenizer

    # Set up trainer
    logger.info("Setting up trainer...")
    training_args = configure_training_args(output_dir, **training_args_kwargs)

    # Create trainer with updated parameters to avoid deprecation warnings
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": preprocessed_dataset["train"],
        "eval_dataset": preprocessed_dataset["validation"],
        "data_collator": data_collator,
        "compute_metrics": compute_metrics,
        "callbacks": [EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
    }

    # Store the tokenizer so we can use it for compute_metrics
    compute_metrics.tokenizer = tokenizer
    trainer_kwargs["tokenizer"] = tokenizer

    # Create the trainer
    trainer = MonetaryAmountTrainer(**trainer_kwargs)

    # Train the model
    logger.info("Starting training...")
    train_result = trainer.train()

    # Log training metrics
    logger.info(f"Training completed. Training metrics: {train_result.metrics}")
    if wandb_logging:
        wandb.log({"train_runtime": train_result.metrics["train_runtime"]})
        wandb.log({"train_samples_per_second": train_result.metrics["train_samples_per_second"]})

    # Evaluate on validation set
    logger.info("Evaluating on validation set...")
    eval_results = trainer.evaluate()
    logger.info(f"Validation metrics: {eval_results}")

    # Save the best model
    best_model_path = os.path.join(output_dir, "best")
    logger.info(f"Saving best model to {best_model_path}")
    save_model(
        trainer.model,
        tokenizer,
        best_model_path,
        metadata={
            "model_name": model_name,
            "train_data_path": str(train_data_path),
            "val_data_path": str(val_data_path),
            "train_metrics": train_result.metrics,
            "eval_metrics": eval_results,
        },
    )

    # Generate training reports
    try:
        logger.info("Generating training reports...")
        generate_training_reports(
            trainer,
            tokenizer,
            preprocessed_dataset["validation"],
            os.path.join(output_dir, "reports"),
        )
    except Exception as e:
        logger.error(f"Failed to generate training reports: {e}")
        logger.error("Continuing without reports generation")

    # Finish W&B run
    if wandb_logging:
        wandb.finish()

    return best_model_path


def generate_training_reports(trainer, tokenizer, eval_dataset, output_dir, num_examples=10):
    """
    Generate training reports including example predictions and metrics.

    Args:
        trainer (MonetaryAmountTrainer): Trained trainer
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer
        eval_dataset (datasets.Dataset): Evaluation dataset
        output_dir (str): Directory to save reports
        num_examples (int): Number of examples to include in the report
    """
    logger.info("Generating training reports...")

    # Get predictions for the entire evaluation dataset
    predictions = trainer.predict(eval_dataset)
    
    # Get metrics
    metrics = predictions.metrics
    
    # Convert predictions and labels to numpy if needed
    preds = predictions.predictions
    labels = predictions.label_ids
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    # Sample indices for the report
    num_examples = min(num_examples, len(eval_dataset))
    example_indices = np.random.choice(len(eval_dataset), num_examples, replace=False)
    
    # Generate report
    report = {
        "metrics": metrics,
        "examples": []
    }
    
    # Add example predictions
    for idx in example_indices:
        example = eval_dataset[idx]
        input_text = example["input"]
        
        # Get predictions for this example
        dollars_pred = float(preds[idx][0])
        cents_pred = float(preds[idx][1])
        dollars_true = float(labels[idx][0])
        cents_true = float(labels[idx][1])
        
        report["examples"].append({
            "input": input_text,
            "prediction": {
                "dollars": dollars_pred,
                "cents": cents_pred,
                "formatted": f"${dollars_pred:.0f}.{cents_pred:.0f}"
            },
            "target": {
                "dollars": dollars_true,
                "cents": cents_true,
                "formatted": f"${dollars_true:.0f}.{cents_true:.0f}"
            }
        })
    
    # Save report
    report_path = os.path.join(output_dir, "training_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Training report saved to {report_path}")
    
    # Log example predictions
    logger.info("\nExample predictions:")
    for example in report["examples"]:
        logger.info(f"\nInput: {example['input']}")
        logger.info(f"Predicted: {example['prediction']['formatted']}")
        logger.info(f"Target: {example['target']['formatted']}")
