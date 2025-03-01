.PHONY: help test test-verbose test-integration test-slow test-all lint lint-fix format clean all generate-data train train-test evaluate inference check-deps update-deps coverage build

# Default target when running 'make' without arguments
help:
	@echo "Available commands:"
	@echo "  make help                 - Show this help message"
	@echo "  make test                 - Run all tests (excluding slow and integration tests)"
	@echo "  make test-verbose         - Run all tests with verbose output"
	@echo "  make test-integration     - Run integration tests"
	@echo "  make test-slow            - Run slow tests"
	@echo "  make test-all             - Run all tests including slow and integration tests"
	@echo "  make coverage             - Generate test coverage report"
	@echo "  make lint                 - Run linter to check code quality"
	@echo "  make lint-fix             - Run linter and fix issues automatically"
	@echo "  make format               - Format code using ruff"
	@echo "  make clean                - Remove temporary files and directories"
	@echo "  make all                  - Run all checks (lint, format, test)"
	@echo "  make check-deps           - Check for outdated dependencies"
	@echo "  make update-deps          - Update dependencies to latest versions"
	@echo "  make build                - Build the package"
	@echo ""
	@echo "Project-specific commands:"
	@echo "  make generate-data        - Generate synthetic training data"
	@echo "  make train                - Train the model"
	@echo "  make train-test           - Run a quick test of the training pipeline"
	@echo "  make evaluate             - Evaluate the model"
	@echo "  make inference            - Run inference with the model"

# Run standard tests (excluding slow and integration tests)
test:
	python -m pytest -m "not slow and not integration"

# Run tests with verbose output
test-verbose:
	python -m pytest -v -m "not slow and not integration"

# Run integration tests
test-integration:
	python -m pytest -v --run-integration -m "integration"

# Run slow tests
test-slow:
	python -m pytest -v --run-slow -m "slow"

# Run all tests including slow and integration tests
test-all:
	python -m pytest -v --run-integration --run-slow

# Generate test coverage report
coverage:
	python -m pytest --cov=src --cov-report=html --cov-report=term

# Run linter to check code quality
lint:
	uv run ruff check .

# Run linter and fix issues automatically
lint-fix:
	uv run ruff check --fix .

# Format code using ruff
format:
	uv run ruff format .

# Run all checks
all: lint format test
	@echo "All checks passed!"

# Generate synthetic training data
generate-data:
	python -m main generate-data

# Train the model
train:
	python -m main train

# Run a quick test of the training pipeline
train-test:
	python -m main train --test-run

# Evaluate the model
evaluate:
	@if [ -z "$(MODEL_PATH)" ]; then \
		echo "ERROR: MODEL_PATH is required. Use 'make evaluate MODEL_PATH=path/to/model'"; \
		exit 1; \
	fi
	python -m main evaluate --model-path $(MODEL_PATH)

# Run inference with the model
inference:
	@if [ -z "$(MODEL_PATH)" ]; then \
		echo "ERROR: MODEL_PATH is required. Use 'make inference MODEL_PATH=path/to/model'"; \
		exit 1; \
	fi
	@if [ -n "$(TEXT)" ]; then \
		python -m main infer --model-path $(MODEL_PATH) --text "$(TEXT)"; \
	elif [ -n "$(INPUT_FILE)" ]; then \
		python -m main infer --model-path $(MODEL_PATH) --input-file $(INPUT_FILE); \
	else \
		python -m main infer --model-path $(MODEL_PATH); \
	fi

# Check for outdated dependencies
check-deps:
	uv pip list --outdated

# Update dependencies to latest versions
update-deps:
	uv pip install --upgrade accelerate datasets evaluate inflect numpy pandas pytest python-levenshtein ruff torch tqdm transformers wandb
	uv pip freeze > requirements.txt

# Build the package
build:
	uv pip install build
	python -m build

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