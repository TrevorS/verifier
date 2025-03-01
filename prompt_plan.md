## Prompts for Implementation

### Prompt 1: Project Setup and Environment

```
Set up a project for a sequence-to-sequence model that converts verbal monetary expressions to JSON. The model will transform phrases like "twenty-five dollars and ten cents" into structured JSON like {"amount": 25.10}.

1. Create a file structure with:
   - main.py (entry point)
   - data/ (for datasets)
   - src/ (for source code)
   - models/ (for saved models)
   - config.py (for configuration)

2. In config.py, define:
   - Model parameters (FLAN-T5-Small)
   - Training parameters (batch size, learning rate, etc.)
   - Paths for data and model saving
   - Logging configuration

3. Set up main.py with:
   - HuggingFace argument parsing
   - Basic logging
   - Command modes for training, evaluation, and inference

4. Ensure the following dependencies have been installed using uv:
   - transformers
   - datasets
   - evaluate
   - torch
   - numpy
   - pandas
   - tqdm
   - wandb
   - python-Levenshtein
   - inflect

The project should use Python 3.12 and support installation with uv. Follow best practices for modular Python code organization.
```

### Prompt 2: Number to Words Utilities

```
Create utility functions for converting numerical monetary amounts to verbal expressions. We'll need these for generating training data.

In src/utils.py, implement:

1. A function to convert numbers to words using the inflect library:
   - Handle dollars (whole numbers)
   - Handle cents (decimal parts)
   - Format properly with "dollars" and "cents"
   - Support variations (with/without "and")
   - Always stick to USD currency

2. Helper functions for text normalization:
   - Convert to lowercase
   - Normalize whitespace
   - Standardize formatting

3. JSON formatting functions:
   - Create proper JSON with amount field
   - Ensure amount has two decimal places
   - Format JSON strings consistently

Implement testing for these functions to verify they work correctly with various inputs.

Examples of desired conversions:
- 25.10 → "twenty-five dollars and ten cents"
- 5.00 → "five dollars"
- 0.75 → "seventy-five cents"
- 100.01 → "one hundred dollars and one cent"

Ensure all text is lowercase and whitespace is normalized in the outputs.
```

### Prompt 3: Synthetic Data Generation

```
Create a data generation module to produce synthetic training data of verbal monetary expressions paired with their JSON representations.

In src/data_generator.py, implement:

1. A function to generate random monetary amounts:
   - Generate random dollar amounts (integers)
   - Generate random cent amounts (0-99)
   - Cover a wide range (e.g., from $0.01 to $1,000,000)
   - Ensure good distribution across different magnitudes

2. A function to create verbal expressions for these amounts:
   - Use the number-to-words function from utils.py
   - Generate format variations as part of the base generation:
     * "twenty dollars" vs "twenty dollars and zero cents"
     * "fifty cents" vs "zero dollars and fifty cents"
     * With and without the word "and"
     * Different ways to express the same amount

3. A function to create the target JSON output:
   - Format: {"amount": 25.10}
   - Ensure amount has two decimal places

4. A main function to generate a dataset:
   - Generate a specified number of examples
   - Ensure diverse coverage of amount ranges
   - Split into training and validation sets
   - Save as JSONL files with input/output pairs

5. Add data augmentation functions:
   - Random character dropout for robustness
   - Case variations (though we'll normalize to lowercase)
   - Other noise that doesn't change the meaning

Generate a small sample dataset and verify the quality of the examples.
```

### Prompt 4: Dataset Preparation with HF Datasets

```
Create dataset processing modules using the HuggingFace datasets library to prepare the data for training.

In src/dataset.py, implement:

1. A function to load JSONL data with datasets:
   - Use datasets.load_dataset with the 'json' format
   - Configure train/validation splits

2. A function to preprocess the dataset:
   - Apply tokenization using the FLAN-T5 tokenizer
   - Handle input and target text formatting
   - Set maximum lengths for inputs and targets
   - Map preprocessing over the dataset

3. A function to configure the FLAN-T5 tokenizer:
   - Load the tokenizer from pretrained
   - Configure for sequence-to-sequence tasks
   - Handle special tokens

4. A data collator for sequence-to-sequence training:
   - Use DataCollatorForSeq2Seq from transformers
   - Configure label padding (-100 for ignored positions)

5. A function to prepare the complete dataset for training:
   - Load data from JSONL
   - Apply preprocessing
   - Configure the data collator
   - Return ready-to-use datasets for training and validation

Test the dataset preparation to ensure tokenization works correctly and the format is compatible with the HuggingFace Trainer.
```

### Prompt 5: Model Configuration with HF Transformers

```
Set up the FLAN-T5-Small model using HuggingFace transformers for the sequence-to-sequence task.

In src/model.py, implement:

1. A function to initialize the FLAN-T5 model:
   - Load the pretrained FLAN-T5-Small model
   - Configure for sequence-to-sequence generation
   - Handle device placement (CPU/GPU)

2. A function to save model checkpoints:
   - Save model with metadata
   - Include tokenizer 
   - Add configuration details

3. A function to load model from checkpoints:
   - Load model and tokenizer
   - Restore configuration
   - Verify model is ready for use

4. A function for text generation:
   - Configure generation parameters (greedy decoding)
   - Generate predictions from input text
   - Convert token IDs back to text

5. Functions to prepare inputs for inference:
   - Tokenize input text
   - Format for model input
   - Handle batch processing if needed

Test the model loading and basic inference on a simple example to verify configuration.
```

### Prompt 6: Training Pipeline with HF Trainer

```
Implement the training pipeline using the HuggingFace Trainer API for fine-tuning FLAN-T5.

In src/trainer.py, implement:

1. A function to configure training arguments:
   - Set up Seq2SeqTrainingArguments
   - Configure learning rate, batch size, and schedule
   - Set evaluation strategy and saving behavior
   - Enable W&B logging
   - Configure early stopping

2. A function to set up the Seq2SeqTrainer:
   - Initialize with model, tokenizer, and datasets
   - Configure data collator
   - Set up compute_metrics function
   - Connect with W&B for experiment tracking

3. Custom evaluation metrics:
   - Exact match accuracy for JSON outputs
   - Levenshtein distance between outputs
   - JSON validity checking
   - Numeric difference between predicted and actual amounts

4. A function to run the training process:
   - Initialize W&B run
   - Train the model
   - Evaluate on validation data
   - Save the best model checkpoint

5. A function to generate training reports:
   - Log final metrics
   - Save example predictions
   - Summarize training results

Update main.py to include a training mode that calls these functions.

Test the training pipeline with a small subset of data to verify it works end-to-end.
```

### Prompt 7: Evaluation and Error Analysis

```
Create a comprehensive evaluation module to assess model performance and analyze errors.

In src/evaluation.py, implement:

1. A function to evaluate model performance on test data:
   - Run inference on test examples
   - Calculate exact match accuracy
   - Calculate numeric difference between predicted and actual amounts
   - Check JSON validity
   - Generate a performance report

2. Functions for error analysis:
   - Categorize errors (formatting, value errors)
   - Parse JSON to extract numerical values
   - Calculate absolute and relative differences in monetary values
   - Identify patterns in incorrect predictions

3. A function to analyze specific examples:
   - Show input, expected output, and model prediction
   - Highlight discrepancies
   - Calculate numeric differences

4. Functions to visualize results:
   - Plot accuracy by amount range
   - Show distribution of numeric errors
   - Create error distribution charts

5. A function to generate a detailed evaluation report:
   - Overall metrics
   - Error breakdown
   - Example predictions (correct and incorrect)
   - Recommendations for improvement

Update main.py to include an evaluation mode that uses these functions.

Test the evaluation module on the model trained in the previous step.
```

### Prompt 8: Inference Pipeline

```
Develop an inference pipeline to use the trained model for converting new verbal monetary expressions to JSON.

In src/inference.py, implement:

1. A function to prepare input for inference:
   - Clean and normalize input text (lowercase, whitespace)
   - Tokenize the input
   - Handle batched inputs if needed

2. A function to run inference with the model:
   - Generate predictions using greedy decoding
   - Decode output tokens to text
   - Handle any generation errors

3. Functions for post-processing model output:
   - Extract JSON from the raw model output
   - Validate JSON structure
   - Parse the amount value
   - Handle malformed JSON gracefully

4. A complete inference pipeline function:
   - Take a verbal monetary expression as input
   - Process through the complete pipeline
   - Return a validated JSON output or None if invalid
   - Include parsed numerical amount

5. A demo script with example inputs:
   - Include varied examples
   - Show complete pipeline execution
   - Display formatted results

Update main.py to include an inference mode that uses this pipeline.

Create a simple CLI interface for testing the inference pipeline with custom inputs.
```
