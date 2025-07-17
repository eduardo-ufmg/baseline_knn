"""Test configuration and fixtures."""

import numpy as np
import pytest
from sklearn.datasets import make_classification


@pytest.fixture  # type: ignore[misc]
def sample_data() -> tuple[np.ndarray, np.ndarray]:
    """Generate sample classification data for testing."""
    X, y = make_classification(
        n_samples=1000, n_features=4, n_informative=3, n_redundant=1, random_state=0
    )
    return X, y


@pytest.fixture  # type: ignore[misc]
def small_data() -> tuple[np.ndarray, np.ndarray]:
    """Generate small dataset for quick tests."""
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([0, 1, 0, 1])
    return X, y
