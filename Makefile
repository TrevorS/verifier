.PHONY: help test lint format clean all generate-data build

NUM_EXAMPLES = 500000
TRAIN_RATIO = 0.9
VAL_RATIO = 0.05
OUTPUT_DIR = data
SEED = 42
AUGMENTATION_RATIO = 0.3
HARD_EXAMPLES_RATIO = 0.05

# Default target when running 'make' without arguments
help:
	@echo "Available commands:"
	@echo "  make help                 - Show this help message"
	@echo "  make test                 - Run all tests (excluding slow and integration tests)"
	@echo "  make lint                 - Run linter to fix code quality"
	@echo "  make format               - Format code using ruff"
	@echo "  make clean                - Remove temporary files and directories"
	@echo "  make all                  - Run all checks (lint, format, test)"
	@echo "  make build                - Build the package"
	@echo ""
	@echo "Project-specific commands:"
	@echo "  make generate-data        - Generate synthetic training data"

# Run standard tests (excluding slow and integration tests)
test:
	@uv run pytest

# Run linter and fix issues automatically
lint:
	@uv run ruff check --fix .

# Format code using ruff
format:
	@uv run ruff format .

# Run all checks
all: lint format test
	@echo "All checks passed!"

generate-data:
	@echo "Generating synthetic training data..."
	@uv run python -m main \
		--num-examples $(NUM_EXAMPLES) \
		--train-ratio $(TRAIN_RATIO) \
		--val-ratio $(VAL_RATIO) \
		--output-dir $(OUTPUT_DIR) \
		--seed $(SEED) \
		--augmentation-ratio $(AUGMENTATION_RATIO) \
		--hard-examples-ratio $(HARD_EXAMPLES_RATIO)

# Build the package
build:
	@uv pip install build
	@uv run python -m build

# Remove temporary files and directories
clean:
	rm -rf .pytest_cache
	rm -rf __pycache__
	rm -rf src/__pycache__
	rm -rf tests/__pycache__
	rm -rf .ruff_cache
	rm -rf .coverage
	rm -rf htmlcov
	rm -rf dist
	rm -rf build
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete