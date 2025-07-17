"""Test configuration and fixtures."""

import pytest
import numpy as np
from sklearn.datasets import make_classification


@pytest.fixture
def sample_data():
    """Generate sample classification data for testing."""
    X, y = make_classification(
        n_samples=100,
        n_features=4,
        n_informative=3,
        n_redundant=1,
        n_clusters_per_class=1,
        random_state=42
    )
    return X, y


@pytest.fixture
def small_data():
    """Generate small dataset for quick tests."""
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([0, 1, 0, 1])
    return X, y
