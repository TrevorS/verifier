.PHONY: help test test-verbose lint lint-fix format clean all generate-data train evaluate inference

# Default target when running 'make' without arguments
help:
	@echo "Available commands:"
	@echo "  make help                 - Show this help message"
	@echo "  make test                 - Run all tests"
	@echo "  make test-verbose         - Run all tests with verbose output"
	@echo "  make lint                 - Run linter to check code quality"
	@echo "  make lint-fix             - Run linter and fix issues automatically"
	@echo "  make format               - Format code using ruff"
	@echo "  make clean                - Remove temporary files and directories"
	@echo "  make all                  - Run all checks (lint, format, test)"
	@echo ""
	@echo "Project-specific commands:"
	@echo "  make generate-data        - Generate synthetic training data"
	@echo "  make train                - Train the model"
	@echo "  make evaluate             - Evaluate the model"
	@echo "  make inference            - Run inference with the model"

# Run all tests
test:
	python -m pytest

# Run tests with verbose output
test-verbose:
	python -m pytest -v

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

# Evaluate the model
evaluate:
	python -m main evaluate

# Run inference with the model
inference:
	python -m main inference

# Remove temporary files and directories
clean:
	rm -rf .pytest_cache
	rm -rf __pycache__
	rm -rf src/__pycache__
	rm -rf tests/__pycache__
	rm -rf .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete 