# Tests for Monetary Expressions Converter

This directory contains tests for the monetary expressions to JSON converter.

## Running Tests

To run all tests:

```bash
python -m pytest
```

To run specific test files:

```bash
python -m pytest tests/test_utils.py
```

To run tests with verbose output:

```bash
python -m pytest -v
```

## Test Coverage

The test suite covers the following functionality:

- `test_utils.py`: Tests for the utility functions in `src/utils.py`:
  - Text normalization
  - JSON formatting
  - Number to words conversion 