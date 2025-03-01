# Verbal Monetary Expression to JSON Converter: Implementation Checklist

## 1. Project Setup and Environment

- [ ] Create project directory structure
  - [ ] Create `main.py` (entry point)
  - [ ] Create `data/` directory (for datasets)
  - [ ] Create `src/` directory (for source code)
  - [ ] Create `models/` directory (for saved models)
  - [ ] Create `config.py` file (for configuration)

- [ ] Set up configuration in `config.py`
  - [ ] Define model parameters (FLAN-T5-Small)
  - [ ] Define training parameters (batch size, learning rate, epochs, etc.)
  - [ ] Define paths for data and model saving
  - [ ] Configure logging parameters
  - [ ] Set random seeds for reproducibility

- [ ] Set up main execution script in `main.py`
  - [ ] Implement HuggingFace argument parsing
  - [ ] Configure basic logging
  - [ ] Define command modes (train, evaluate, infer)
  - [ ] Add argument validation
  - [ ] Set up main execution flow

- [ ] Create environment setup files
  - [ ] Create `requirements.txt` with all dependencies
  - [ ] Document Python 3.12 requirement
  - [ ] Add installation instructions for uv
  - [ ] Create `.gitignore` file

- [ ] Configure Weights & Biases integration
  - [ ] Set up project initialization
  - [ ] Configure default logging parameters

## 2. Number to Words Utilities

- [ ] Create `src/utils.py` for utility functions
  - [ ] Implement function to convert numbers to words using inflect
    - [ ] Handle dollars (whole numbers)
    - [ ] Handle cents (decimal parts)
    - [ ] Format with "dollars" and "cents"
    - [ ] Support variations (with/without "and")
  
  - [ ] Implement text normalization functions
    - [ ] Convert text to lowercase
    - [ ] Normalize whitespace
    - [ ] Standardize formatting

  - [ ] Implement JSON formatting functions
    - [ ] Create JSON with amount field
    - [ ] Ensure amount has two decimal places
    - [ ] Format JSON strings consistently

- [ ] Create unit tests for utility functions
  - [ ] Test number to words conversion
    - [ ] Test with whole dollar amounts
    - [ ] Test with cents only
    - [ ] Test with dollars and cents
    - [ ] Test with zero cents
    - [ ] Test with large numbers
  
  - [ ] Test text normalization
    - [ ] Test lowercase conversion
    - [ ] Test whitespace normalization
  
  - [ ] Test JSON formatting
    - [ ] Test proper decimal formatting
    - [ ] Test structural correctness

## 3. Synthetic Data Generation

- [ ] Create `src/data_generator.py` for data generation
  - [ ] Implement function to generate random monetary amounts
    - [ ] Generate random dollar amounts
    - [ ] Generate random cent amounts
    - [ ] Ensure diverse range coverage
    - [ ] Control distribution across magnitudes

  - [ ] Implement function to create verbal expressions
    - [ ] Use number-to-words function from utils.py
    - [ ] Generate format variations:
      - [ ] With/without "and"
      - [ ] With/without "zero cents"
      - [ ] Just cents for amounts < $1
      - [ ] Different ways to express the same amount

  - [ ] Implement function to create target JSON output
    - [ ] Format as `{"amount": 25.10}`
    - [ ] Ensure consistent decimal places

  - [ ] Implement dataset generation function
    - [ ] Generate specified number of examples
    - [ ] Ensure diverse amount ranges
    - [ ] Split into training, validation, and test sets
    - [ ] Save as JSONL files

  - [ ] Implement data augmentation functions
    - [ ] Random character dropout
    - [ ] Add noise that doesn't change meaning
    - [ ] Apply augmentation to a portion of the dataset

- [ ] Create scripts to generate and verify datasets
  - [ ] Create script to generate full training dataset
  - [ ] Create script to generate evaluation dataset
  - [ ] Add data quality verification
  - [ ] Generate sample sets for testing

## 4. Dataset Preparation with HF Datasets

- [ ] Create `src/dataset.py` for dataset handling
  - [ ] Implement function to load JSONL data
    - [ ] Use datasets.load_dataset with 'json' format
    - [ ] Configure train/validation splits
    - [ ] Add caching configuration

  - [ ] Implement tokenizer configuration
    - [ ] Load FLAN-T5 tokenizer
    - [ ] Configure for sequence-to-sequence tasks
    - [ ] Handle special tokens
    - [ ] Set maximum lengths

  - [ ] Implement dataset preprocessing function
    - [ ] Apply tokenization to input and target
    - [ ] Format texts consistently
    - [ ] Apply text normalization
    - [ ] Map preprocessing over dataset

  - [ ] Set up data collator
    - [ ] Use DataCollatorForSeq2Seq
    - [ ] Configure label padding
    - [ ] Set up dynamic padding

  - [ ] Create complete dataset preparation function
    - [ ] Load data from JSONL
    - [ ] Apply preprocessing
    - [ ] Configure data collator
    - [ ] Return prepared datasets

- [ ] Create test script for dataset verification
  - [ ] Verify tokenization works correctly
  - [ ] Check dataset formatting
  - [ ] Ensure compatibility with HF Trainer
  - [ ] Test dataset loading speed

## 5. Model Configuration with HF Transformers

- [ ] Create `src/model.py` for model handling
  - [ ] Implement function to initialize FLAN-T5
    - [ ] Load pretrained FLAN-T5-Small
    - [ ] Configure for sequence-to-sequence
    - [ ] Handle device placement (CPU/GPU)
    - [ ] Set up model configuration

  - [ ] Implement model checkpoint saving
    - [ ] Save model with metadata
    - [ ] Include tokenizer
    - [ ] Add configuration details
    - [ ] Implement versioning

  - [ ] Implement model loading from checkpoints
    - [ ] Load model and tokenizer
    - [ ] Restore configuration
    - [ ] Verify model is ready for use
    - [ ] Handle different checkpoint formats

  - [ ] Implement text generation function
    - [ ] Configure greedy decoding
    - [ ] Generate predictions from input
    - [ ] Convert tokens back to text
    - [ ] Handle generation errors

  - [ ] Implement input preparation for inference
    - [ ] Tokenize input text
    - [ ] Format for model input
    - [ ] Handle batch processing

- [ ] Create test script for model verification
  - [ ] Test model initialization
  - [ ] Test saving and loading
  - [ ] Test basic inference
  - [ ] Verify output formatting

## 6. Training Pipeline with HF Trainer

- [ ] Create `src/trainer.py` for training pipeline
  - [ ] Implement training arguments configuration
    - [ ] Set up Seq2SeqTrainingArguments
    - [ ] Configure learning rate and schedule
    - [ ] Set batch size
    - [ ] Configure evaluation strategy
    - [ ] Set up checkpoint saving
    - [ ] Enable W&B logging
    - [ ] Configure early stopping

  - [ ] Implement Seq2SeqTrainer setup
    - [ ] Initialize with model and datasets
    - [ ] Configure data collator
    - [ ] Set up compute_metrics function
    - [ ] Connect with W&B tracking

  - [ ] Implement custom evaluation metrics
    - [ ] Exact match accuracy
    - [ ] Levenshtein distance
    - [ ] JSON validity check
    - [ ] Numeric difference calculation

  - [ ] Implement training execution function
    - [ ] Initialize W&B run
    - [ ] Train the model
    - [ ] Evaluate on validation
    - [ ] Save best checkpoint
    - [ ] Handle training interruptions

  - [ ] Implement training report generation
    - [ ] Log final metrics
    - [ ] Save example predictions
    - [ ] Generate training summary
    - [ ] Create visualizations

- [ ] Update main.py with training mode
  - [ ] Add training command
  - [ ] Configure training parameters
  - [ ] Add argument parsing for training
  - [ ] Implement training workflow

- [ ] Create test script for training pipeline
  - [ ] Test with small dataset
  - [ ] Verify metrics calculation
  - [ ] Check checkpoint saving
  - [ ] Test W&B integration

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
