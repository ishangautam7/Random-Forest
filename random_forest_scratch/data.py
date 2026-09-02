"""Wine dataset loading and a dependency-free stratified train/test split."""

from __future__ import annotations

from pathlib import Path
import hashlib
import urllib.request

import numpy as np

FEATURE_NAMES = (
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
)
DATASET_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/"
    "winequality-red.csv"
)
DATASET_SHA256 = "4a402cf041b025d4566d954c3b9ba8635a3a8a01e039005d97d6a710278cf05e"
SOURCE_ROW_COUNT = 1599
MODELING_ROW_COUNT = 1359
DUPLICATE_ROWS_REMOVED = SOURCE_ROW_COUNT - MODELING_ROW_COUNT


def download_wine_data(destination: str | Path) -> Path:
    """Download the official UCI file and verify its SHA-256 digest."""

    output_path = Path(destination)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(DATASET_URL, timeout=30) as response:
        content = response.read()

    digest = hashlib.sha256(content).hexdigest()
    if digest != DATASET_SHA256:
        raise RuntimeError(
            "Downloaded dataset checksum did not match the expected official file"
        )
    output_path.write_bytes(content)
    # Validate the content before reporting success.
    load_wine_data(output_path)
    return output_path


def load_wine_data(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load and validate UCI's red Wine Quality CSV file."""

    data = np.loadtxt(path, delimiter=";", skiprows=1, dtype=float)
    if data.ndim != 2 or data.shape[1] != 12:
        raise ValueError("Expected 12 columns: 11 features followed by wine quality")
    if len(data) != SOURCE_ROW_COUNT:
        raise ValueError(
            f"Expected {SOURCE_ROW_COUNT} rows from the UCI dataset, found {len(data)}"
        )
    if not np.all(np.isfinite(data)):
        raise ValueError("Dataset contains missing or non-finite values")

    # The source contains 240 exact duplicate records. Keeping identical copies
    # on both sides of a random split would make test performance look better
    # than it really is, so retain only each record's first occurrence.
    _, unique_indices = np.unique(data, axis=0, return_index=True)
    data = data[np.sort(unique_indices)]
    if len(data) != MODELING_ROW_COUNT:
        raise ValueError(
            f"Expected {MODELING_ROW_COUNT} unique rows, found {len(data)}"
        )

    X = data[:, :-1]
    y = data[:, -1].astype(np.int64)
    if set(np.unique(y)) != {3, 4, 5, 6, 7, 8}:
        raise ValueError("Expected observed wine-quality scores 3 through 8")
    return X, y


def stratified_train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    *,
    test_size: float = 0.2,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split arrays while retaining approximately equal class proportions."""

    X_array = np.asarray(X)
    y_array = np.asarray(y)
    if X_array.ndim != 2 or y_array.ndim != 1 or len(X_array) != len(y_array):
        raise ValueError("X must be 2D and y must be a matching 1D array")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be in (0, 1)")

    rng = np.random.default_rng(random_state)
    train_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    for label in np.unique(y_array):
        indices = np.flatnonzero(y_array == label)
        rng.shuffle(indices)
        n_test = min(len(indices) - 1, max(1, int(round(len(indices) * test_size))))
        test_parts.append(indices[:n_test])
        train_parts.append(indices[n_test:])

    train_indices = np.concatenate(train_parts)
    test_indices = np.concatenate(test_parts)
    rng.shuffle(train_indices)
    rng.shuffle(test_indices)
    return (
        X_array[train_indices],
        X_array[test_indices],
        y_array[train_indices],
        y_array[test_indices],
    )
