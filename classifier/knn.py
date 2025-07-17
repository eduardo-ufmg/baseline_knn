"""
Baseline KNN Classifier.

This module implements a baseline k-Nearest Neighbors (k-NN) classifier
that automatically tunes its hyperparameters using Bayesian Optimization.
"""

import numpy as np
import warnings
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from skopt import BayesSearchCV
from skopt.space import Categorical, Integer

# It's good practice to set a seed for reproducibility
np.random.seed(0)


class BaselineKNNClassifier(BaseEstimator, ClassifierMixin):  # type: ignore[misc]
    """
    A non-parametric k-Nearest Neighbors (k-NN) classifier.

    This class wraps scikit-learn's KNeighborsClassifier and uses BayesSearchCV
    from scikit-optimize to find the optimal combination of n_neighbors, weights,
    and the p-metric during the fitting process.

    Parameters
    ----------
    n_iter : int, default=8
        The number of parameter settings that are sampled. This is the number
        of iterations for the Bayesian Optimization search.

    cv : int, default=3
        The number of folds to use for cross-validation during the
        optimization process.

    random_state : int, default=0
        Seed used by the random number generator for reproducibility of the
        optimization search.
    """

    def __init__(self, n_iter: int = 8, cv: int = 3, random_state: int = 0) -> None:
        """Initialize the BaselineKNNClassifier with hyperparameter search settings."""
        self.n_iter = n_iter
        self.cv = cv
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaselineKNNClassifier":
        """
        Find the best hyperparameters for k-NN using Bayesian Optimization.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples,)
            The target values.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # 1. Check and validate input data
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)  # Store the classes seen during fit

        # 2. Define the base estimator and the hyperparameter search space
        knn = KNeighborsClassifier()

        # NOTE: kNN is already a non-parametric model. This class makes it
        # "hyperparameter-free" from the user's perspective by automating tuning.

        # Calculate the maximum safe n_neighbors based on dataset size and CV
        # With k-fold CV, the minimum training set size is approximately
        # (k-1)/k * n_samples
        min_train_size = int((self.cv - 1) / self.cv * len(X))
        # Leave some safety margin and ensure at least n_neighbors=1
        max_neighbors = max(1, min(50, min_train_size - 1))

        search_spaces = {
            "n_neighbors": Integer(1, max_neighbors),
            "weights": Categorical(["uniform", "distance"]),
            "p": Integer(1, 2),  # 1 for Manhattan distance, 2 for Euclidean
        }

        # 3. Set up the Bayesian Optimization search with cross-validation
        # We use StratifiedKFold to maintain the percentage of samples for each class.
        cv_splitter = StratifiedKFold(
            n_splits=self.cv, shuffle=True, random_state=self.random_state
        )

        opt = BayesSearchCV(
            estimator=knn,
            search_spaces=search_spaces,
            n_iter=self.n_iter,
            cv=cv_splitter,
            scoring="accuracy",
            random_state=self.random_state,
            verbose=0,
        )

        # 4. Run the optimization on the training data, suppressing user warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            opt.fit(X, y)

        # 5. Store the best found estimator and its parameters
        self.best_estimator_ = opt.best_estimator_
        self.best_params_ = opt.best_params_
        self.best_score_ = opt.best_score_

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the class labels for the provided data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples to predict.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Class labels for each data sample.
        """
        # Check if fit has been called
        check_is_fitted(self)
        # Validate the input
        X = check_array(X)
        # Delegate prediction to the best found estimator
        return self.best_estimator_.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return probability estimates for the test data X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples to predict.

        Returns
        -------
        p : ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples.
        """
        # Check if fit has been called
        check_is_fitted(self)
        # Validate the input
        X = check_array(X)
        # Delegate probability prediction to the best found estimator
        return self.best_estimator_.predict_proba(X)
