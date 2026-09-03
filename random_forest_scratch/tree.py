from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Literal

import numpy as np

MaxFeatures = int | float | Literal["sqrt", "log2"] | None


@dataclass(slots=True)
class _TreeNode:
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
    def __init__(
        self,
        *,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: MaxFeatures = None,
        random_state: int | None = None,
    ) -> None:
        self.max_depth = int(max_depth) if max_depth is not None else None
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_features = max_features
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> DecisionTreeClassifier:
        X_array = np.asarray(X, dtype=float)
        y_array = np.asarray(y)
        classes, y_encoded = np.unique(y_array, return_inverse=True)
        return self._fit_encoded(X_array, y_encoded, len(classes), classes)

    def _fit_encoded(
        self,
        X: np.ndarray,
        y_encoded: np.ndarray,
        n_classes: int,
        classes: np.ndarray,
    ) -> DecisionTreeClassifier:
        self.n_classes_ = n_classes
        self.classes_ = classes
        self._max_features_count = _resolve_max_features(self.max_features, X.shape[1])
        self._rng = np.random.default_rng(self.random_state)
        self._raw_feature_importances = np.zeros(X.shape[1], dtype=float)
        self.root_ = self._grow_tree(X, y_encoded, depth=0)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[self._predict_encoded(X)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_array = np.asarray(X, dtype=float)
        probabilities = np.empty((len(X_array), self.n_classes_), dtype=float)
        for row_index, row in enumerate(X_array):
            node = self._leaf_for_row(row)
            probabilities[row_index] = node.class_counts / node.class_counts.sum()
        return probabilities

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y)))

    def _grow_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> _TreeNode:
        counts = np.bincount(y, minlength=self.n_classes_)
        node = _TreeNode(predicted_class=int(np.argmax(counts)), class_counts=counts)

        if (self.max_depth is not None and depth >= self.max_depth) or (
            len(y) < self.min_samples_split
            or len(y) < 2 * self.min_samples_leaf
            or np.count_nonzero(counts) == 1
        ):
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
        best_gain = 0.0
        best_split = None

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

                if gain > best_gain:
                    best_gain = gain
                    threshold = (values[split_index] + values[split_index + 1]) / 2
                    best_split = int(feature_index), float(threshold), gain

        return best_split

    def _predict_encoded(self, X: np.ndarray) -> np.ndarray:
        X_array = np.asarray(X, dtype=float)
        return np.array(
            [self._leaf_for_row(row).predicted_class for row in X_array]
        )

    def _leaf_for_row(self, row: np.ndarray) -> _TreeNode:
        node = self.root_
        while not node.is_leaf:
            feature_index = node.feature_index
            threshold = node.threshold
            assert feature_index is not None and threshold is not None
            next_node = node.left if row[feature_index] <= threshold else node.right
            assert next_node is not None
            node = next_node
        return node


def _gini_from_counts(counts: np.ndarray) -> float:
    proportions = counts / counts.sum()
    return float(1.0 - np.dot(proportions, proportions))


def _resolve_max_features(max_features: MaxFeatures, n_features: int) -> int:
    if max_features is None:
        return n_features
    if max_features == "sqrt":
        return max(1, int(np.sqrt(n_features)))
    if max_features == "log2":
        return max(1, int(np.log2(n_features)))
    if isinstance(max_features, Integral):
        return int(max_features)
    return max(1, int(np.ceil(float(max_features) * n_features)))
