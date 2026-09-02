import numpy as np
import pytest

from random_forest_scratch import DecisionTreeClassifier


def test_tree_learns_simple_boundary() -> None:
    X = np.arange(20, dtype=float).reshape(-1, 1)
    y = np.where(X[:, 0] < 10, "low", "high")
    tree = DecisionTreeClassifier(max_depth=2, random_state=1).fit(X, y)

    assert tree.score(X, y) == 1.0
    assert tree.predict(np.array([[2.5], [15.5]])).tolist() == ["low", "high"]
    np.testing.assert_allclose(tree.predict_proba(X).sum(axis=1), 1.0)


def test_tree_rejects_wrong_feature_count() -> None:
    tree = DecisionTreeClassifier().fit(np.array([[0.0], [1.0]]), np.array([0, 1]))
    with pytest.raises(ValueError, match="expected 1"):
        tree.predict(np.array([[0.0, 1.0]]))

