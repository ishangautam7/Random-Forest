"""Evaluation helpers for multiclass classification."""

from __future__ import annotations

import numpy as np


def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a multiclass confusion matrix and its ordered labels."""

    true = np.asarray(y_true)
    predicted = np.asarray(y_pred)
    if true.ndim != 1 or predicted.ndim != 1 or len(true) != len(predicted):
        raise ValueError("y_true and y_pred must be matching one-dimensional arrays")
    ordered_labels = (
        np.unique(np.concatenate((true, predicted)))
        if labels is None
        else np.asarray(labels)
    )
    if ordered_labels.ndim != 1 or len(ordered_labels) == 0:
        raise ValueError("labels must be a non-empty one-dimensional array")
    if not set(np.unique(np.concatenate((true, predicted)))).issubset(
        set(ordered_labels.tolist())
    ):
        raise ValueError("labels must include every value in y_true and y_pred")

    matrix = np.zeros((len(ordered_labels), len(ordered_labels)), dtype=np.int64)
    label_to_index = {label: index for index, label in enumerate(ordered_labels)}
    for actual, predicted_label in zip(true, predicted):
        matrix[label_to_index[actual], label_to_index[predicted_label]] += 1
    return matrix, ordered_labels


def multiclass_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: np.ndarray | None = None,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """Calculate accuracy and macro-averaged precision, recall, and F1."""

    true = np.asarray(y_true)
    predicted = np.asarray(y_pred)
    matrix, ordered_labels = confusion_matrix(true, predicted, labels)
    precisions: list[float] = []
    recalls: list[float] = []
    f1_scores: list[float] = []

    def safe_divide(numerator: float, denominator: float) -> float:
        return float(numerator / denominator) if denominator else 0.0

    for index in range(len(ordered_labels)):
        true_positive = int(matrix[index, index])
        false_positive = int(matrix[:, index].sum() - true_positive)
        false_negative = int(matrix[index, :].sum() - true_positive)
        precision = safe_divide(true_positive, true_positive + false_positive)
        recall = safe_divide(true_positive, true_positive + false_negative)
        f1 = safe_divide(2 * precision * recall, precision + recall)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    return {
        "accuracy": float(np.mean(true == predicted)),
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1_scores)),
        "within_one_accuracy": float(np.mean(np.abs(true - predicted) <= 1)),
        "mean_absolute_error": float(np.mean(np.abs(true - predicted))),
    }, matrix, ordered_labels
