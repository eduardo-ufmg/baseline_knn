# baseline_knn

The baseline KNN against which I compare the ones I study.

## Overview

This repository provides a baseline K-Nearest Neighbors (KNN) implementation designed for comparison studies. It follows modern Python packaging standards and includes comprehensive profiling tools for memory and speed analysis.

## Features

- Modern Python packaging with `pyproject.toml`
- Pre-commit hooks for code quality (black, isort, flake8, mypy)
- Comprehensive testing with pytest
- Memory and speed profiling tools
- Benchmarking utilities
- Type hints and strict typing
- Development workflow automation

## Project Structure

```
baseline_knn/
├── src/baseline_knn/          # Main package source code
│   └── __init__.py
├── tests/                     # Test suite
│   ├── conftest.py           # Test configuration and fixtures
│   └── test_knn.py           # KNN implementation tests
├── profiling/                 # Profiling tools and utilities
│   ├── memory_profiler.py    # Memory profiling utilities
│   ├── speed_profiler.py     # Speed profiling utilities
│   ├── profile_runner.py     # Comprehensive profiling runner
│   └── README.md             # Profiling documentation
├── pyproject.toml            # Project configuration and dependencies
├── .pre-commit-config.yaml   # Pre-commit hooks configuration
├── Makefile                  # Development workflow automation
├── LICENSE
└── README.md
```

## Installation

### Basic Installation
```bash
pip install -e .
```

### Development Installation
```bash
pip install -e ".[dev]"
```

### With Profiling Tools
```bash
pip install -e ".[profiling,benchmark]"
```

### Full Installation (recommended for development)
```bash
pip install -e ".[dev,profiling,benchmark]"
```

## Development Workflow

### Setup Development Environment
```bash
make dev-setup
```

### Code Formatting
```bash
make format          # Format code with black and isort
make format-check    # Check formatting without changes
```

### Linting and Type Checking
```bash
make lint           # Run flake8 and mypy
```

### Testing
```bash
make test           # Run tests
make test-cov       # Run tests with coverage
```

### Pre-commit Hooks
```bash
make pre-commit     # Install and run pre-commit hooks
```

### Profiling
```bash
make profile        # Run comprehensive profiling
make profile-memory # Run memory profiling only
make profile-speed  # Run speed profiling only
make benchmark      # Run benchmarking
```

### All Checks
```bash
make check          # Run formatting check, linting, and tests
```

## Profiling Tools

This project includes comprehensive profiling tools for performance analysis:

- **Memory Profiling**: Track memory usage with multiple tools (memory_profiler, memray, scalene)
- **Speed Profiling**: Analyze execution time with cProfile and line_profiler
- **Benchmarking**: Compare multiple implementations with statistical analysis

See `profiling/README.md` for detailed documentation.

## Configuration

The project uses `pyproject.toml` for all configuration:

- **Build system**: setuptools with modern configuration
- **Dependencies**: Core, development, and profiling dependencies
- **Tools**: black, isort, mypy, pytest, coverage configuration
- **Metadata**: Project information and classifiers

## Dependencies

### Core Dependencies
- numpy: Numerical computing
- scikit-learn: Machine learning library
- pandas: Data manipulation

### Development Dependencies
- black: Code formatting
- isort: Import sorting
- flake8: Linting
- mypy: Type checking
- pytest: Testing framework
- pre-commit: Git hooks

### Profiling Dependencies
- memory-profiler: Line-by-line memory usage
- line-profiler: Line-by-line timing
- memray: Advanced memory profiling
- py-spy: Sampling profiler
- scalene: CPU, memory, and GPU profiler

## Contributing

1. Install development dependencies: `make install-dev`
2. Set up pre-commit hooks: `make pre-commit`
3. Make your changes
4. Run checks: `make check`
5. Submit a pull request

## License

See LICENSE file for details.
