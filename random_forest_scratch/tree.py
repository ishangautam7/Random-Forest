"""CART-style classification tree used by the Random Forest.

This module intentionally implements the learning algorithm directly instead of
wrapping scikit-learn. NumPy is used only for arrays and numerical operations.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Literal

import numpy as np

MaxFeatures = int | float | Literal["sqrt", "log2"] | None


@dataclass(slots=True)
class _TreeNode:
    """One node in a binary decision tree."""

    predicted_class: int
    class_counts: np.ndarray
    feature_index: int | None = None
    threshold: float | None = None
    left: _TreeNode | None = None
    right: _TreeNode | None = None

    @property
    def is_leaf(self) -> bool:
        return self.feature_index is None


class DecisionTreeClassifier:
    """A decision tree classifier trained with greedy Gini-impurity splits.

    Parameters mirror the most useful parts of common ML libraries, but every
    split, node, and prediction is implemented in this project.
    """

    def __init__(
        self,
        *,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: MaxFeatures = None,
        min_impurity_decrease: float = 0.0,
        random_state: int | None = None,
    ) -> None:
        if max_depth is not None and (not isinstance(max_depth, Integral) or max_depth < 1):
            raise ValueError("max_depth must be None or a positive integer")
        if not isinstance(min_samples_split, Integral) or min_samples_split < 2:
            raise ValueError("min_samples_split must be an integer >= 2")
        if not isinstance(min_samples_leaf, Integral) or min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be an integer >= 1")
        if min_impurity_decrease < 0:
            raise ValueError("min_impurity_decrease must be >= 0")

        self.max_depth = int(max_depth) if max_depth is not None else None
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_features = max_features
        self.min_impurity_decrease = float(min_impurity_decrease)
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> DecisionTreeClassifier:
        """Fit the tree and return ``self``."""

        X_array, y_array = _validate_training_data(X, y)
        classes, y_encoded = np.unique(y_array, return_inverse=True)
        self.classes_ = classes
        return self._fit_encoded(X_array, y_encoded, len(classes), classes)

    def _fit_encoded(
        self,
        X: np.ndarray,
        y_encoded: np.ndarray,
        n_classes: int,
        classes: np.ndarray | None = None,
    ) -> DecisionTreeClassifier:
        """Fit already-encoded targets (used internally by the forest)."""

        self.n_features_in_ = X.shape[1]
        self.n_classes_ = n_classes
        self.classes_ = np.arange(n_classes) if classes is None else classes
        self._max_features_count = _resolve_max_features(self.max_features, X.shape[1])
        self._rng = np.random.default_rng(self.random_state)
        self._raw_feature_importances = np.zeros(X.shape[1], dtype=float)
        self.root_ = self._grow_tree(X, y_encoded, depth=0)

        importance_sum = self._raw_feature_importances.sum()
        if importance_sum > 0:
            self.feature_importances_ = self._raw_feature_importances / importance_sum
        else:
            self.feature_importances_ = self._raw_feature_importances.copy()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the class label for each row."""

        encoded = self._predict_encoded(X)
        return self.classes_[encoded]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class proportions from the leaf reached by each row."""

        X_array = self._validate_prediction_data(X)
        probabilities = np.empty((len(X_array), self.n_classes_), dtype=float)
        for row_index, row in enumerate(X_array):
            node = self._leaf_for_row(row)
            probabilities[row_index] = node.class_counts / node.class_counts.sum()
        return probabilities

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return mean classification accuracy."""

        y_array = np.asarray(y)
        if y_array.ndim != 1:
            raise ValueError("y must be a one-dimensional array")
        predictions = self.predict(X)
        if len(predictions) != len(y_array):
            raise ValueError("X and y must contain the same number of samples")
        return float(np.mean(predictions == y_array))

    def _grow_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> _TreeNode:
        counts = np.bincount(y, minlength=self.n_classes_)
        node = _TreeNode(predicted_class=int(np.argmax(counts)), class_counts=counts)

        reached_depth_limit = self.max_depth is not None and depth >= self.max_depth
        cannot_split = (
            len(y) < self.min_samples_split
            or len(y) < 2 * self.min_samples_leaf
            or np.count_nonzero(counts) == 1
        )
        if reached_depth_limit or cannot_split:
            return node

        split = self._best_split(X, y, counts)
        if split is None:
            return node

        feature_index, threshold, gain = split
        left_mask = X[:, feature_index] <= threshold
        node.feature_index = feature_index
        node.threshold = threshold
        node.left = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._grow_tree(X[~left_mask], y[~left_mask], depth + 1)

        # CART importance: samples reaching the node multiplied by Gini gain.
        self._raw_feature_importances[feature_index] += len(y) * gain
        return node

    def _best_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        parent_counts: np.ndarray,
    ) -> tuple[int, float, float] | None:
        n_samples, n_features = X.shape
        parent_impurity = _gini_from_counts(parent_counts)
        best_gain = self.min_impurity_decrease
        best_feature: int | None = None
        best_threshold: float | None = None

        feature_indices = self._rng.choice(
            n_features, size=self._max_features_count, replace=False
        )
        for feature_index in feature_indices:
            order = np.argsort(X[:, feature_index], kind="mergesort")
            values = X[order, feature_index]
            labels = y[order]
            left_counts = np.zeros(self.n_classes_, dtype=np.int64)

            for split_index in range(n_samples - 1):
                left_counts[labels[split_index]] += 1
                left_size = split_index + 1
                right_size = n_samples - left_size

                if left_size < self.min_samples_leaf or right_size < self.min_samples_leaf:
                    continue
                if values[split_index] == values[split_index + 1]:
                    continue

                right_counts = parent_counts - left_counts
                weighted_impurity = (
                    left_size * _gini_from_counts(left_counts)
                    + right_size * _gini_from_counts(right_counts)
                ) / n_samples
                gain = parent_impurity - weighted_impurity

                if gain > best_gain + 1e-12:
                    best_gain = gain
                    best_feature = int(feature_index)
                    # This form is safer than (a + b) / 2 for large numbers.
                    lower = values[split_index]
                    upper = values[split_index + 1]
                    best_threshold = float(lower + (upper - lower) / 2.0)

        if best_feature is None or best_threshold is None:
            return None
        return best_feature, best_threshold, best_gain

    def _predict_encoded(self, X: np.ndarray) -> np.ndarray:
        X_array = self._validate_prediction_data(X)
        return np.fromiter(
            (self._leaf_for_row(row).predicted_class for row in X_array),
            dtype=np.int64,
            count=len(X_array),
        )

    def _leaf_for_row(self, row: np.ndarray) -> _TreeNode:
        node = self.root_
        while not node.is_leaf:
            if row[node.feature_index] <= node.threshold:  # type: ignore[index,operator]
                node = node.left  # type: ignore[assignment]
            else:
                node = node.right  # type: ignore[assignment]
        return node

    def _validate_prediction_data(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "root_"):
            raise RuntimeError("This DecisionTreeClassifier has not been fitted")
        X_array = np.asarray(X, dtype=float)
        if X_array.ndim != 2:
            raise ValueError("X must be a two-dimensional array")
        if X_array.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X_array.shape[1]} features; expected {self.n_features_in_}"
            )
        if not np.all(np.isfinite(X_array)):
            raise ValueError("X must contain only finite values")
        return X_array


def _gini_from_counts(counts: np.ndarray) -> float:
    total = counts.sum()
    if total == 0:
        return 0.0
    proportions = counts / total
    return float(1.0 - np.dot(proportions, proportions))


def _resolve_max_features(max_features: MaxFeatures, n_features: int) -> int:
    if max_features is None:
        return n_features
    if max_features == "sqrt":
        return max(1, int(np.sqrt(n_features)))
    if max_features == "log2":
        return max(1, int(np.log2(n_features)))
    if isinstance(max_features, Integral) and not isinstance(max_features, bool):
        if not 1 <= int(max_features) <= n_features:
            raise ValueError(f"integer max_features must be in [1, {n_features}]")
        return int(max_features)
    if isinstance(max_features, Real) and not isinstance(max_features, bool):
        if not 0 < float(max_features) <= 1:
            raise ValueError("float max_features must be in (0, 1]")
        return max(1, int(np.ceil(float(max_features) * n_features)))
    raise ValueError("max_features must be None, 'sqrt', 'log2', an int, or a float")


def _validate_training_data(
    X: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    X_array = np.asarray(X, dtype=float)
    y_array = np.asarray(y)
    if X_array.ndim != 2:
        raise ValueError("X must be a two-dimensional array")
    if y_array.ndim != 1:
        raise ValueError("y must be a one-dimensional array")
    if len(X_array) != len(y_array):
        raise ValueError("X and y must contain the same number of samples")
    if len(X_array) == 0 or X_array.shape[1] == 0:
        raise ValueError("X and y cannot be empty")
    if not np.all(np.isfinite(X_array)):
        raise ValueError("X must contain only finite values")
    return X_array, y_array

