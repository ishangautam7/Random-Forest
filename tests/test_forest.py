import numpy as np

from random_forest_scratch import RandomForestClassifier
from random_forest_scratch.data import load_wine_data, stratified_train_test_split


def make_separable_data(seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(240, 4))
    y = ((X[:, 0] + 0.6 * X[:, 1]) > 0).astype(int)
    return X, y


def test_forest_is_accurate_and_probabilities_are_votes() -> None:
    X, y = make_separable_data()
    forest = RandomForestClassifier(
        n_estimators=31, max_depth=7, random_state=42
    ).fit(X, y)

    assert forest.score(X, y) > 0.95
    probabilities = forest.predict_proba(X[:8])
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    # 31 hard votes means every probability is a multiple of 1/31.
    np.testing.assert_allclose(probabilities * 31, np.round(probabilities * 31))
    assert 0.0 <= forest.oob_score_ <= 1.0
    assert forest.oob_coverage_ > 0.95


def test_forest_is_reproducible_for_a_fixed_seed() -> None:
    X, y = make_separable_data()
    first = RandomForestClassifier(n_estimators=9, random_state=123).fit(X, y)
    second = RandomForestClassifier(n_estimators=9, random_state=123).fit(X, y)

    np.testing.assert_array_equal(first.predict(X), second.predict(X))
    np.testing.assert_allclose(first.feature_importances_, second.feature_importances_)
    assert first.oob_score_ == second.oob_score_


def test_stratified_split_keeps_both_classes() -> None:
    X, y = make_separable_data()
    X_train, X_test, y_train, y_test = stratified_train_test_split(
        X, y, test_size=0.25, random_state=5
    )

    assert len(X_train) + len(X_test) == len(X)
    assert set(y_train) == {0, 1}
    assert set(y_test) == {0, 1}


def test_forest_supports_multiple_classes() -> None:
    rng = np.random.default_rng(11)
    X = np.concatenate(
        [rng.normal(loc=value, scale=0.2, size=(30, 2)) for value in (-2, 0, 2)]
    )
    y = np.repeat([3, 5, 8], 30)
    forest = RandomForestClassifier(n_estimators=15, max_depth=4, random_state=4).fit(X, y)

    assert forest.score(X, y) > 0.95
    assert forest.predict_proba(X[:3]).shape == (3, 3)


def test_wine_loader_removes_exact_duplicates() -> None:
    X, y = load_wine_data("data/winequality-red.csv")

    assert len(X) == len(y) == 1359
    assert len(np.unique(np.column_stack((X, y)), axis=0)) == 1359
