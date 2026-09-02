import numpy as np

from random_forest_scratch.metrics import multiclass_classification_metrics


def test_multiclass_metrics() -> None:
    truth = np.array([3, 4, 5, 5, 6, 6])
    predicted = np.array([3, 5, 5, 6, 6, 6])

    scores, matrix, labels = multiclass_classification_metrics(truth, predicted)

    assert scores["accuracy"] == 4 / 6
    assert scores["within_one_accuracy"] == 1.0
    assert scores["mean_absolute_error"] == 2 / 6
    assert matrix.sum() == 6
    assert labels.tolist() == [3, 4, 5, 6]
