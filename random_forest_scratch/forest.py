from __future__ import annotations

from numbers import Integral, Real

import numpy as np

from .tree import DecisionTreeClassifier, MaxFeatures, _validate_training_data


class RandomForestClassifier:
    def __init__(
        self,
        *,
        n_estimators: int = 100,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: MaxFeatures = "sqrt",
        bootstrap: bool = True,
        max_samples: int | float | None = None,
        oob_score: bool = True,
        random_state: int | None = None,
    ) -> None:
        if not isinstance(n_estimators, Integral) or n_estimators < 1:
            raise ValueError("n_estimators must be a positive integer")
        if oob_score and not bootstrap:
            raise ValueError("oob_score requires bootstrap=True")

        self.n_estimators = int(n_estimators)
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.max_samples = max_samples
        self.oob_score = oob_score
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
        X_array, y_array = _validate_training_data(X, y)
        self.classes_, y_encoded = np.unique(y_array, return_inverse=True)
        self.n_classes_ = len(self.classes_)
        self.n_features_in_ = X_array.shape[1]
        sample_size = self._resolve_sample_size(len(X_array))
        rng = np.random.default_rng(self.random_state)

        self.estimators_: list[DecisionTreeClassifier] = []
        importance_total = np.zeros(self.n_features_in_, dtype=float)
        if self.oob_score:
            oob_votes = np.zeros((len(X_array), self.n_classes_), dtype=np.int64)
            oob_counts = np.zeros(len(X_array), dtype=np.int64)

        for _ in range(self.n_estimators):
            tree_seed = int(rng.integers(0, np.iinfo(np.int32).max))
            if self.bootstrap:
                sample_indices = rng.integers(0, len(X_array), size=sample_size)
            else:
                sample_indices = rng.choice(len(X_array), size=sample_size, replace=False)

            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=tree_seed,
            )
            tree._fit_encoded(
                X_array[sample_indices],
                y_encoded[sample_indices],
                self.n_classes_,
                self.classes_,
            )
            self.estimators_.append(tree)
            importance_total += tree._raw_feature_importances

            if self.oob_score:
                selected = np.zeros(len(X_array), dtype=bool)
                selected[np.unique(sample_indices)] = True
                oob_indices = np.flatnonzero(~selected)
                if len(oob_indices):
                    predictions = tree._predict_encoded(X_array[oob_indices])
                    oob_votes[oob_indices, predictions] += 1
                    oob_counts[oob_indices] += 1

        total_importance = importance_total.sum()
        self.feature_importances_ = (
            importance_total / total_importance
            if total_importance > 0
            else importance_total
        )

        if self.oob_score:
            valid = oob_counts > 0
            self.oob_decision_function_ = np.full(
                (len(X_array), self.n_classes_), np.nan, dtype=float
            )
            self.oob_decision_function_[valid] = (
                oob_votes[valid] / oob_counts[valid, np.newaxis]
            )
            self.oob_score_ = float(
                np.mean(np.argmax(oob_votes[valid], axis=1) == y_encoded[valid])
            )
            self.oob_coverage_ = float(np.mean(valid))

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        votes = self._collect_votes(X)
        return self.classes_[np.argmax(votes, axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        votes = self._collect_votes(X)
        return votes / self.n_estimators

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y_array = np.asarray(y)
        predictions = self.predict(X)
        if y_array.ndim != 1 or len(y_array) != len(predictions):
            raise ValueError("y must be one-dimensional and match X's row count")
        return float(np.mean(predictions == y_array))

    def _collect_votes(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "estimators_"):
            raise RuntimeError("This RandomForestClassifier has not been fitted")
        X_array = np.asarray(X, dtype=float)
        if X_array.ndim != 2:
            raise ValueError("X must be a two-dimensional array")
        if X_array.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X_array.shape[1]} features; expected {self.n_features_in_}"
            )
        if not np.all(np.isfinite(X_array)):
            raise ValueError("X must contain only finite values")

        votes = np.zeros((len(X_array), self.n_classes_), dtype=np.int64)
        row_indices = np.arange(len(X_array))
        for tree in self.estimators_:
            predictions = tree._predict_encoded(X_array)
            votes[row_indices, predictions] += 1
        return votes

    def _resolve_sample_size(self, n_samples: int) -> int:
        if self.max_samples is None:
            return n_samples
        if isinstance(self.max_samples, Integral) and not isinstance(self.max_samples, bool):
            if not 1 <= int(self.max_samples) <= n_samples:
                raise ValueError(f"integer max_samples must be in [1, {n_samples}]")
            return int(self.max_samples)
        if isinstance(self.max_samples, Real) and not isinstance(self.max_samples, bool):
            if not 0 < float(self.max_samples) <= 1:
                raise ValueError("float max_samples must be in (0, 1]")
            return max(1, int(round(float(self.max_samples) * n_samples)))
        raise ValueError("max_samples must be None, an int, or a float")
