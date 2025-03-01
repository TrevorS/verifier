# Verbal Monetary Expression to JSON Converter: Implementation Checklist

## 1. Project Setup and Environment

- [x] Create project directory structure
  - [x] Create `main.py` (entry point)
  - [x] Create `data/` directory (for datasets)
  - [x] Create `src/` directory (for source code)
  - [x] Create `models/` directory (for saved models)
  - [x] Create `config.py` file (for configuration)

- [x] Set up configuration in `config.py`
  - [x] Define model parameters (FLAN-T5-Small)
  - [x] Define training parameters (batch size, learning rate, epochs, etc.)
  - [x] Define paths for data and model saving
  - [x] Configure logging parameters
  - [x] Set random seeds for reproducibility

- [x] Set up main execution script in `main.py`
  - [x] Implement HuggingFace argument parsing
  - [x] Configure basic logging
  - [x] Define command modes (train, evaluate, infer)
  - [x] Add argument validation
  - [x] Set up main execution flow

- [x] Create environment setup files
  - [x] Create `requirements.txt` with all dependencies
  - [x] Document Python 3.12 requirement
  - [x] Add installation instructions for uv
  - [x] Create `.gitignore` file

- [x] Configure Weights & Biases integration
  - [x] Set up project initialization
  - [x] Configure default logging parameters

## 2. Number to Words Utilities

- [x] Create `src/utils.py` for utility functions
  - [x] Implement function to convert numbers to words using inflect
    - [x] Handle dollars (whole numbers)
    - [x] Handle cents (decimal parts)
    - [x] Format with "dollars" and "cents"
    - [x] Support variations (with/without "and")
  
  - [x] Implement text normalization functions
    - [x] Convert text to lowercase
    - [x] Normalize whitespace
    - [x] Standardize formatting

  - [x] Implement JSON formatting functions
    - [x] Create JSON with amount field
    - [x] Ensure amount has two decimal places
    - [x] Format JSON strings consistently

- [x] Create unit tests for utility functions
  - [x] Test number to words conversion
    - [x] Test with whole dollar amounts
    - [x] Test with cents only
    - [x] Test with dollars and cents
    - [x] Test with zero cents
    - [x] Test with large numbers
  
  - [x] Test text normalization
    - [x] Test lowercase conversion
    - [x] Test whitespace normalization
  
  - [x] Test JSON formatting
    - [x] Test proper decimal formatting
    - [x] Test structural correctness

## 3. Synthetic Data Generation

- [x] Create `src/data_generator.py` for data generation
  - [x] Implement function to generate random monetary amounts
    - [x] Generate random dollar amounts
    - [x] Generate random cent amounts
    - [x] Ensure diverse range coverage
    - [x] Control distribution across magnitudes

  - [x] Implement function to create verbal expressions
    - [x] Use number-to-words function from utils.py
    - [x] Generate format variations:
      - [x] With/without "and"
      - [x] With/without "zero cents"
      - [x] Just cents for amounts < $1
      - [x] Different ways to express the same amount

  - [x] Implement function to create target JSON output
    - [x] Format as `{"amount": 25.10}`
    - [x] Ensure consistent decimal places

  - [x] Implement dataset generation function
    - [x] Generate specified number of examples
    - [x] Ensure diverse amount ranges
    - [x] Split into training, validation, and test sets
    - [x] Save as JSONL files

  - [x] Implement data augmentation functions
    - [x] Random character dropout
    - [x] Add noise that doesn't change meaning
    - [x] Apply augmentation to a portion of the dataset

- [x] Create scripts to generate and verify datasets
  - [x] Create script to generate full training dataset
  - [x] Create script to generate evaluation dataset
  - [x] Add data quality verification
  - [x] Generate sample sets for testing

## 4. Dataset Preparation with HF Datasets

- [x] Create `src/dataset.py` for dataset handling
  - [x] Implement function to load JSONL data
    - [x] Use datasets.load_dataset with 'json' format
    - [x] Configure train/validation splits
    - [x] Add caching configuration

  - [x] Implement tokenizer configuration
    - [x] Load FLAN-T5 tokenizer
    - [x] Configure for sequence-to-sequence tasks
    - [x] Handle special tokens
    - [x] Set maximum lengths

  - [x] Implement dataset preprocessing function
    - [x] Apply tokenization to input and target
    - [x] Format texts consistently
    - [x] Apply text normalization
    - [x] Map preprocessing over dataset

  - [x] Set up data collator
    - [x] Use DataCollatorForSeq2Seq
    - [x] Configure label padding
    - [x] Set up dynamic padding

  - [x] Create complete dataset preparation function
    - [x] Load data from JSONL
    - [x] Apply preprocessing
    - [x] Configure data collator
    - [x] Return prepared datasets

- [x] Create test script for dataset verification
  - [x] Verify tokenization works correctly
  - [x] Check dataset formatting
  - [x] Ensure compatibility with HF Trainer
  - [x] Test dataset loading speed

## 5. Model Configuration with HF Transformers

- [x] Create `src/model.py` for model handling
  - [x] Implement function to initialize FLAN-T5
    - [x] Load pretrained FLAN-T5-Small
    - [x] Configure for sequence-to-sequence
    - [x] Handle device placement (CPU/GPU)
    - [x] Set up model configuration

  - [x] Implement model checkpoint saving
    - [x] Save model with metadata
    - [x] Include tokenizer
    - [x] Add configuration details
    - [x] Implement versioning

  - [x] Implement model loading from checkpoints
    - [x] Load model and tokenizer
    - [x] Restore configuration
    - [x] Verify model is ready for use
    - [x] Handle different checkpoint formats

  - [x] Implement text generation function
    - [x] Configure greedy decoding
    - [x] Generate predictions from input
    - [x] Convert tokens back to text
    - [x] Handle generation errors

  - [x] Implement input preparation for inference
    - [x] Tokenize input text
    - [x] Format for model input
    - [x] Handle batch processing

- [x] Create test script for model verification
  - [x] Test model initialization
  - [x] Test saving and loading
  - [x] Test basic inference
  - [x] Verify output formatting

## 6. Training Pipeline with HF Trainer

- [x] Create `src/trainer.py` for training pipeline
  - [x] Implement training arguments configuration
    - [x] Set up Seq2SeqTrainingArguments
    - [x] Configure learning rate and schedule
    - [x] Set batch size
    - [x] Configure evaluation strategy
    - [x] Set up checkpoint saving
    - [x] Enable W&B logging
    - [x] Configure early stopping

  - [x] Implement Seq2SeqTrainer setup
    - [x] Initialize with model and datasets
    - [x] Configure data collator
    - [x] Set up compute_metrics function
    - [x] Connect with W&B tracking

  - [x] Implement custom evaluation metrics
    - [x] Exact match accuracy
    - [x] Levenshtein distance
    - [x] JSON validity check
    - [x] Numeric difference calculation

  - [x] Implement training execution function
    - [x] Initialize W&B run
    - [x] Train the model
    - [x] Evaluate on validation
    - [x] Save best checkpoint
    - [x] Handle training interruptions

  - [x] Implement training report generation
    - [x] Log final metrics
    - [x] Save example predictions
    - [x] Generate training summary
    - [x] Create visualizations

- [x] Update main.py with training mode
  - [x] Add training command
  - [x] Configure training parameters
  - [x] Add argument parsing for training
  - [x] Implement training workflow

- [x] Create test script for training pipeline
  - [x] Test with small dataset
  - [x] Verify metrics calculation
  - [x] Check checkpoint saving
  - [x] Test W&B integration

## 7. Evaluation and Error Analysis

- [ ] Create `src/evaluation.py` for evaluation
  - [ ] Implement model evaluation function
    - [ ] Run inference on test data
    - [ ] Calculate exact match accuracy
    - [ ] Calculate numeric differences
    - [ ] Check JSON validity
    - [ ] Generate performance report

  - [ ] Implement error analysis functions
    - [ ] Categorize errors by type
    - [ ] Parse JSON to extract values
    - [ ] Calculate absolute/relative differences
    - [ ] Identify error patterns
    - [ ] Generate error statistics

  - [ ] Implement example analysis function
    - [ ] Show input and outputs
    - [ ] Highlight discrepancies
    - [ ] Calculate differences
    - [ ] Provide context for errors

  - [ ] Implement visualization functions
    - [ ] Plot accuracy by amount range
    - [ ] Show numeric error distribution
    - [ ] Create error category charts
    - [ ] Generate comparison visualizations

  - [ ] Implement evaluation report generation
    - [ ] Compile overall metrics
    - [ ] Create error breakdown
    - [ ] Include example predictions
    - [ ] Add improvement recommendations

- [ ] Update main.py with evaluation mode
  - [ ] Add evaluation command
  - [ ] Configure evaluation parameters
  - [ ] Add argument parsing for evaluation
  - [ ] Implement evaluation workflow

- [ ] Create test script for evaluation module
  - [ ] Test with sample predictions
  - [ ] Verify metric calculations
  - [ ] Test report generation
  - [ ] Check visualization functions

## 8. Inference Pipeline

- [ ] Create `src/inference.py` for inference
  - [ ] Implement input preparation function
    - [ ] Clean and normalize text
    - [ ] Tokenize the input
    - [ ] Handle batch inputs
    - [ ] Validate input format

  - [ ] Implement model inference function
    - [ ] Configure greedy decoding
    - [ ] Generate predictions
    - [ ] Decode tokens to text
    - [ ] Handle generation errors

  - [ ] Implement output post-processing
    - [ ] Extract JSON from output
    - [ ] Validate JSON structure
    - [ ] Parse amount value
    - [ ] Handle malformed JSON
    - [ ] Format final output

  - [ ] Implement complete inference pipeline
    - [ ] Process input through all steps
    - [ ] Return validated JSON or error
    - [ ] Include parsed amount
    - [ ] Add confidence information

  - [ ] Create demo script with examples
    - [ ] Include varied test cases
    - [ ] Show complete execution
    - [ ] Display formatted results
    - [ ] Demonstrate error handling

- [ ] Update main.py with inference mode
  - [ ] Add inference command
  - [ ] Configure inference parameters
  - [ ] Add argument parsing
  - [ ] Implement inference workflow

- [ ] Create CLI interface for testing
  - [ ] Accept custom inputs
  - [ ] Display formatted results
  - [ ] Allow batch processing
  - [ ] Provide help documentation

## 9. Documentation and Final Review

- [ ] Create comprehensive README.md
  - [ ] Project overview
  - [ ] Installation instructions
  - [ ] Usage examples
  - [ ] Configuration options
  - [ ] Training documentation
  - [ ] Evaluation metrics explanation

- [ ] Document code with docstrings
  - [ ] Add docstrings to all functions
  - [ ] Include parameter descriptions
  - [ ] Document return values
  - [ ] Add usage examples where helpful

- [ ] Create example notebooks
  - [ ] Data generation tutorial
  - [ ] Training workflow
  - [ ] Evaluation and analysis
  - [ ] Inference examples

- [ ] Perform final code review
  - [ ] Check for code quality
  - [ ] Verify error handling
  - [ ] Test edge cases
  - [ ] Ensure consistent style
  - [ ] Check for performance issues

- [ ] Review overall project
  - [ ] Verify all components are integrated
  - [ ] Check for feature completeness
  - [ ] Test end-to-end workflows
  - [ ] Document any known limitations
  - [ ] Plan for future improvements
