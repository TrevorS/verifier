# Verbal Monetary Expression to JSON Converter: Project Specification

## Project Overview

### Purpose

This project aims to develop a sequence-to-sequence model that converts verbally expressed monetary amounts (e.g., "twenty-five dollars and ten cents") into a structured JSON format. This enables consistent, programmatic handling of monetary amounts expressed in natural language, such as from speech transcripts or text documents.

### Key Objectives & Output

The model should accurately parse varied verbal expressions of US currency into a JSON object with a clearly defined structure. The expected output is a JSON representation containing the monetary value in a standardized form.

Example:
- Input: "one hundred twenty dollars and fifty cents"
- Output: `{"amount": 120.50}`

The JSON includes the numerical amount for downstream use. High accuracy and proper JSON formatting are critical – the model should produce correctly formatted JSON strings for each input phrase.

## Technical Requirements

### Environment and Dependencies

- Python 3.12
- Package management with `uv`
- Key libraries:
  - HuggingFace transformers (for model)
  - HuggingFace datasets (for data handling)
  - HuggingFace evaluate (for evaluation)
  - HuggingFace Trainer (for training)
  - torch (for underlying computation)
  - numpy, pandas (for data processing)
  - tqdm (for progress tracking)
  - wandb (for experiment tracking)
  - python-Levenshtein (for string similarity metrics)
  - inflect (for number to word conversion)

### Architecture & Model Choice

#### Model Selection: FLAN-T5-Small

We will use FLAN-T5-Small, an 80M-parameter variant of Google's T5 model fine-tuned on instructions. FLAN-T5 has been instruction-tuned on a wide range of tasks which improves its ability to understand and generate structured outputs. The "Small" version is lightweight enough for efficient training and deployment, yet powerful enough for our sequence transformation task.

T5's sequence-to-sequence architecture is well-suited for converting input text to a desired output text (in our case, JSON). By leveraging FLAN-T5-Small, we benefit from its pre-trained knowledge and instruction-following ability, which should help in learning the mapping from verbalized numbers to structured format with less data and training time.

#### Training Approach with HuggingFace Trainer

The model will be fine-tuned using HuggingFace's Trainer API, specifically the Seq2SeqTrainer for sequence-to-sequence tasks. This provides a high-level training loop with support for distributed training, mixed precision, and easy logging.

We will define training arguments and utilize the Trainer to handle forward passes, loss computation, gradient updates, and checkpointing. Using Seq2SeqTrainer ensures that the decoder generates the target JSON text given the input text. It also simplifies integration with evaluation metrics and logging.

### Tokenization & Preprocessing

We will use the FLAN-T5 tokenizer to encode input phrases and decode model outputs. Each verbalized monetary amount (input) is tokenized into subwords/pieces, and likewise the JSON output string is tokenized for the decoder. Key preprocessing steps include:

- **Format Normalization**: Input texts will be cleaned and standardized (lowercased, whitespace normalized) so that the model focuses on essential content. We'll ensure consistent formatting in outputs (including two decimal places for cents in the JSON) during training.

- **Tokenization**: The tokenizer will convert the cleaned input text into input IDs and the expected JSON string into label IDs. We'll set appropriate max_length (for input) and max_target_length (for output) to accommodate the longest expected phrases and JSON.

- **Special Tokens**: The T5 model has designated padding tokens and end-of-sequence tokens. We'll enable padding so that batches of sequences are aligned in length.

By tokenizing inputs/outputs, we transform the problem into a typical sequence-to-sequence learning task where the model predicts a sequence of tokens (the JSON) from another sequence of tokens (the verbal description).

## Data Handling

### Synthetic Data Generation Strategy

Because a large, labeled dataset of verbalized monetary amounts with corresponding structured values may not be readily available, we will generate a synthetic dataset to train the model. We'll programmatically create diverse examples of monetary expressions in English and pair them with the correct JSON.

The synthetic generation will cover a wide range of scenarios, including:

- **Varied Numeric Ranges**: Small amounts (e.g., "fifty cents"), medium amounts ("nineteen dollars and ninety-nine cents"), and larger amounts ("five thousand two hundred dollars") to ensure the model handles different magnitudes and multi-part numbers.

- **Format Variations**: Some expressions might omit certain words or use different phrasings. We'll include such variations (with and without the word "and", with or without the minor unit if it's zero, etc.) as part of our base data generation so the model learns to handle implicit zeros or implied decimals.

- **Currency Simplification**: We'll focus exclusively on USD, simplifying the output to just the amount field without currency specification.

Along with synthetic data, we'll incorporate real examples when available. We'll hold out about 50% of real examples for testing, and the rest can be blended into training. The real data ensures our model sees genuine phrasing patterns that might not be captured by programmatic generation.

### Data Augmentation Techniques

To improve robustness and generalization, we'll apply augmentation on the training data:

- **Random Character Dropout**: On some fraction of synthetic inputs, we'll randomly delete characters to introduce minor noise that simulates typos or transcription errors, forcing the model to learn to handle imperfect input.

- **Other Noise**: We may introduce other forms of noise that don't change the meaning of the expression but help make the model more robust.

Augmentation will only introduce noise that doesn't belong in the original dataset, whereas format variations will be part of the base data generation.

### Data Format and Processing

- All data will be stored in JSONL format, with each line containing a JSON object with the input text and target output.

- We'll use the HuggingFace datasets library for data loading and processing, which provides efficient data handling and transformation operations.

- Before feeding data to the model, we'll perform preprocessing steps including text normalization (lowercase, whitespace normalization) and tokenization.

## Training Pipeline

### Hyperparameters

We'll fine-tune FLAN-T5-Small with carefully chosen hyperparameters:

- **Batch Size**: A batch size of 32 examples per iteration (adjustable based on memory constraints).

- **Learning Rate**: An initial learning rate of 5e-5 for the AdamW optimizer, with potential adjustment based on monitoring.

- **Warmup Steps**: A learning rate warmup of about 500 steps (or ~5% of training steps) at the start of training to avoid shocking the model.

- **Learning Rate Decay**: A linear decay of the learning rate after warmup to ensure convergence.

- **Epochs**: Training for about 3-5 epochs over the dataset, with potential early stopping based on validation metrics.

### Training Configuration

We'll use HuggingFace's Seq2SeqTrainingArguments to configure the training process:

- Enable gradient accumulation if needed for memory constraints
- Use mixed precision training for efficiency
- Save model checkpoints at regular intervals
- Enable early stopping based on validation metrics
- Configure logging to Weights & Biases

### Monitoring and Evaluation During Training

We'll integrate Weights & Biases (W&B) for experiment tracking and monitoring. All training and evaluation metrics will be logged to an interactive W&B dashboard in real-time.

Throughout training, we'll monitor key metrics:
- Training loss
- Validation loss
- Validation exact-match accuracy
- JSON validity
- Numeric accuracy of extracted amounts

If validation metrics stop improving or start deteriorating, we'll halt training early or adjust parameters in subsequent runs.

## Evaluation & Error Analysis

### Evaluation Metrics

To quantitatively assess model performance, we'll use several metrics:

- **Exact Match Accuracy**: The percentage of test examples where the model's JSON output exactly matches the expected JSON string.

- **Field-Level Accuracy**: The accuracy of the amount value, comparing the numerical value in the model's output with the reference value.

- **Levenshtein Distance**: String edit distance between predicted and actual outputs.

- **Numeric Difference**: The absolute and relative differences between predicted and actual monetary amounts after parsing.

We'll test the model on a held-out test set that includes real-world examples and challenging synthetic examples to ensure we measure performance in realistic scenarios.

### Error Breakdown and Analysis

We'll develop a script to analyze incorrect predictions and categorize the errors:

- **Formatting Errors**: The model output is not valid JSON (missing braces, quotes, etc.).

- **Value Errors**: The JSON format is correct, but the numeric amount is wrong.

- **Partial Understanding**: The model captured some parts correctly but not others.

- **Out-of-Scope Input**: Test inputs with patterns not covered in training.

The error analysis script will output counts or percentages of each error type, helping prioritize improvements.

### Iterative Refinement

Our strategy will be iterative: after the first training round, we'll evaluate on the held-out real dataset and analyze errors. Using insights from the error breakdown, we'll refine the training process and data:

- We may augment the synthetic data generation to include cases that address observed errors.
- We may adjust hyperparameters if needed based on error patterns.

After refining, we'll retrain the model on the improved dataset and re-evaluate. We may go through multiple cycles of this process, each time aiming to reduce specific error categories.

## Inference Pipeline

For inference, we'll use the trained model to convert new verbal inputs to JSON:

### Inference Process

1. **Input Preparation**: Clean and normalize input text (lowercase, whitespace normalization).

2. **Model Inference**: Generate predictions using greedy decoding (for deterministic output).

3. **JSON Extraction**: Extract the JSON object from the model's output, handling any formatting issues.

4. **Validation**: Validate the JSON structure and parse the amount value.

5. **Error Handling**: Return None or an error indicator if the model fails to produce valid JSON.

### Post-Processing

After extracting the JSON string, we'll:
- Attempt to parse it using standard JSON parsing
- Verify the presence of the expected fields
- Convert the amount to a numeric value for use in calculations

If parsing fails or the JSON is missing fields, the inference function will return None, indicating that the model could not produce a valid output.

## Testing Plan

To ensure reliable performance before deployment, we'll implement a comprehensive testing plan:

### Final Evaluation on Held-Out Data

After training and refinements, we'll use the held-out real dataset as a final test set to evaluate the model. This dataset has never been seen during training, providing an unbiased estimate of real-world performance.

We'll compute the exact match accuracy, field-level accuracies, and numeric difference metrics on this set. Our target is a high exact-match score (ideally > 95%) on this real data.

### Automated Validation Script

We'll implement an automated test script that:
- Loads the fine-tuned model and tokenizer
- Processes a test file of input phrases with expected outputs
- Compares predictions to expected outputs
- Computes summary metrics and error categorization
- Logs results for inspection

This script ensures that any future changes to the model or data can be validated consistently.

## Future Considerations

While not part of the initial implementation, future enhancements could include:

- Extending the model to handle multi-sentence inputs or amounts embedded in longer text
- Adding support for additional currencies
- Implementing active learning to continuously improve the model with real-world data
- Creating a lightweight API service for integration with other systems

This specification provides a clear blueprint to implement and refine a model for converting verbalized monetary amounts to structured JSON, focusing on accuracy, robustness, and practical usability.
