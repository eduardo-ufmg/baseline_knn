# Makefile for baseline-knn project

.PHONY: help install install-dev install-profiling test lint format clean profile benchmark docs

# Default target
help:
	@echo "Available targets:"
	@echo "  install           - Install package in development mode"
	@echo "  install-dev       - Install with development dependencies"
	@echo "  install-profiling - Install with profiling dependencies"
	@echo "  test              - Run tests"
	@echo "  lint              - Run linting (flake8, mypy)"
	@echo "  format            - Format code (black, isort)"
	@echo "  clean             - Clean build artifacts"
	@echo "  profile           - Run profiling test"
	@echo "  benchmark         - Run benchmark test"
	@echo "  pre-commit        - Install and run pre-commit hooks"
	@echo "  docs              - Generate documentation"

# Installation targets
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

install-profiling:
	pip install -e ".[profiling,benchmark]"

install-all:
	pip install -e ".[dev,profiling,benchmark]"

# Development targets
test:
	pytest -v

test-cov:
	pytest --cov=baseline_knn --cov-report=html --cov-report=term

lint:
	flake8 src/ tests/ profiling/
	mypy src/

format:
	black src/ tests/ profiling/
	isort src/ tests/ profiling/

format-check:
	black --check src/ tests/ profiling/
	isort --check-only src/ tests/ profiling/

# Pre-commit
pre-commit:
	pre-commit install
	pre-commit run --all-files

# Profiling targets
profile:
	python profiling/profile_runner.py --test

profile-memory:
	python profiling/memory_profiler.py --test

profile-speed:
	python profiling/speed_profiler.py --test

benchmark:
	python profiling/speed_profiler.py --benchmark

# Cleanup
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf coverage_html_report/
	rm -rf profiling/results/
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

# Documentation (placeholder)
docs:
	@echo "Documentation generation not yet implemented"

# Development workflow
dev-setup: install-all pre-commit
	@echo "Development environment setup complete!"

check: format-check lint test
	@echo "All checks passed!"
