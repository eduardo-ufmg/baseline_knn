# Baseline KNN Classifier

A comprehensive k-Nearest Neighbors (k-NN) classifier implementation with automated hyperparameter tuning using Bayesian Optimization. This project serves as a baseline for comparing different k-NN variants and implementations.

## 🚀 Features

- **Automated Hyperparameter Tuning**: Uses Bayesian Optimization via `scikit-optimize` to find optimal parameters
- **Comprehensive Evaluation**: Cross-validation experiments with detailed performance metrics
- **Dataset Management**: Tools for downloading, preprocessing, and validating datasets from multiple sources
- **Memory and Time Profiling**: Tracks both CPU time and memory usage during training and prediction
- **Parallel Processing**: Efficient parallel execution for large-scale experiments
- **sklearn Compatible**: Follows sklearn's BaseEstimator and ClassifierMixin interfaces

## 📦 Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Install Dependencies

```bash
pip install -e .
```

For development dependencies:
```bash
pip install -e ".[dev]"
```

## 🔧 Usage

### Basic Usage

```python
from classifier.knn import BaselineKNNClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load sample data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Create and train the classifier
clf = BaselineKNNClassifier(n_iter=32, cv=5, random_state=0)
clf.fit(X_train, y_train)

# Make predictions
predictions = clf.predict(X_test)
probabilities = clf.predict_proba(X_test)

# Access the best parameters found
print(f"Best parameters: {clf.best_params_}")
print(f"Best CV score: {clf.best_score_}")
```

### Running Experiments

The project includes a comprehensive experiment runner that evaluates the classifier on multiple datasets:

```bash
# Download and preprocess datasets
python store_datasets/store_sets.py ./datasets

# Run cross-validation experiments
python experiment.py ./datasets ./results/experiment_results.csv
```

### Dataset Management

The `store_datasets` module provides tools for managing datasets from various sources:

```python
# Download datasets from sklearn, OpenML, and UCI repositories
python store_datasets/store_sets.py ./my_datasets

# Validate processed datasets
python store_datasets/test.py ./my_datasets
```

## 🏗️ Project Structure

```
baseline_knn/
├── classifier/
│   ├── __init__.py
│   └── knn.py              # Main BaselineKNNClassifier implementation
├── store_datasets/         # Dataset management tools (git submodule)
│   ├── store_sets.py       # Download and preprocess datasets
│   ├── test.py            # Validate datasets
│   └── requirements.txt
├── tests/
│   ├── conftest.py        # Test fixtures and configuration
│   └── test_knn.py        # Comprehensive test suite
├── results/               # Experiment results
├── sets/                  # Dataset storage
├── experiment.py          # Main experiment runner
├── pyproject.toml         # Project configuration
└── README.md
```

## 🧪 BaselineKNNClassifier

The core classifier automatically optimizes the following hyperparameters:

- **n_neighbors**: Number of neighbors (1 to min(50, safe_max))
- **weights**: Weighting scheme ('uniform' or 'distance')
- **p**: Distance metric (1 for Manhattan, 2 for Euclidean)

### Parameters

- `n_iter` (int, default=8): Number of optimization iterations
- `cv` (int, default=3): Number of cross-validation folds
- `random_state` (int, default=0): Random seed for reproducibility

### Attributes (after fitting)

- `best_estimator_`: The best KNeighborsClassifier found
- `best_params_`: Dictionary of best hyperparameters
- `best_score_`: Best cross-validation score achieved
- `classes_`: Array of class labels

## 📊 Experiment Results

The experiment runner provides detailed metrics:

- **Accuracy**: Mean and standard deviation across CV folds
- **Timing**: CPU time for fitting and prediction
- **Memory**: Peak memory usage during training and inference
- **Preprocessing**: Automatic feature scaling, variance filtering, and PCA

Results are saved in CSV format with columns:
- `dataset`: Dataset name
- `accuracy_mean/std`: Classification accuracy statistics
- `fit_time_cpu_mean/std`: Training time statistics
- `predict_time_cpu_mean/std`: Prediction time statistics
- `fit_mem_peak_mean_mb/std_mb`: Memory usage statistics
- `predict_mem_peak_mean_mb/std_mb`: Prediction memory statistics

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=classifier

# Run specific test file
pytest tests/test_knn.py
```

### Test Coverage

The test suite includes:
- Parameter validation and edge cases
- Fit/predict functionality
- Input validation and error handling
- Reproducibility tests
- Performance benchmarks
- sklearn compatibility checks

## 🛠️ Development

### Code Quality

This project uses several tools to maintain code quality:

```bash
# Format code
black .

# Sort imports
isort .

# Lint code
flake8 .

# Type checking
mypy .

# Security scanning
bandit -c pyproject.toml -r .
```

### Pre-commit Hooks

Install pre-commit hooks to automatically run quality checks:

```bash
pre-commit install
```

## 📈 Performance Characteristics

- **Time Complexity**: O(n_iter × cv × n_samples × n_features) for training
- **Space Complexity**: O(n_samples × n_features) for storage
- **Scalability**: Efficient parallel processing for multiple datasets
- **Memory Profiling**: Tracks peak memory usage during operations

## 🔍 Dataset Sources

The project supports datasets from:
- **sklearn**: Built-in datasets (iris, wine, breast_cancer, etc.)
- **OpenML**: Community machine learning datasets
- **UCI**: University of California Irvine ML repository

All datasets are automatically:
- Cleaned and preprocessed
- Split into features and targets
- Validated for quality
- Stored in standardized CSV format

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Clone the repository
git clone https://github.com/your-username/baseline_knn.git
cd baseline_knn

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

**Eduardo Henrique Basilio de Carvalho**
- Email: eduardohbc@ufmg.com
- University: UFMG (Universidade Federal de Minas Gerais)

## 🙏 Acknowledgments

- [scikit-learn](https://scikit-learn.org/) for the base k-NN implementation
- [scikit-optimize](https://scikit-optimize.github.io/) for Bayesian optimization
- [OpenML](https://www.openml.org/) and [UCI ML Repository](https://archive.ics.uci.edu/ml/) for datasets

## 📚 Citation

If you use this baseline in your research, please cite:

```bibtex
@software{baseline_knn,
  author = {Eduardo Henrique Basilio de Carvalho},
  title = {Baseline KNN Classifier},
  url = {https://github.com/your-username/baseline_knn},
  version = {0.1.0},
  year = {2025}
}
```
