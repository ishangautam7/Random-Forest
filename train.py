"""Train and evaluate the from-scratch Random Forest on UCI red-wine data."""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from random_forest_scratch import RandomForestClassifier
from random_forest_scratch.data import (
    DUPLICATE_ROWS_REMOVED,
    FEATURE_NAMES,
    SOURCE_ROW_COUNT,
    download_wine_data,
    load_wine_data,
    stratified_train_test_split,
)
from random_forest_scratch.metrics import multiclass_classification_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Random Forest implemented from scratch."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/winequality-red.csv"),
        help="path to the UCI data file (downloaded automatically if absent)",
    )
    parser.add_argument("--trees", type=int, default=100, help="number of trees")
    parser.add_argument("--max-depth", type=int, default=4, help="maximum tree depth")
    parser.add_argument(
        "--min-samples-leaf",
        type=int,
        default=5,
        help="minimum rows in a leaf (regularizes the trees)",
    )
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    return parser.parse_args()


def create_evaluation_plot(
    confusion_matrix: np.ndarray,
    class_labels: np.ndarray,
    importances: np.ndarray,
    output_path: Path,
) -> None:
    """Save a confusion matrix and feature-importance chart."""

    fig, (axis_matrix, axis_importance) = plt.subplots(1, 2, figsize=(13, 5.5))

    image = axis_matrix.imshow(confusion_matrix, cmap="Blues")
    axis_matrix.set_title("Test-set confusion matrix")
    axis_matrix.set_xlabel("Predicted quality")
    axis_matrix.set_ylabel("Actual quality")
    positions = np.arange(len(class_labels))
    axis_matrix.set_xticks(positions, labels=class_labels)
    axis_matrix.set_yticks(positions, labels=class_labels)
    threshold = confusion_matrix.max() / 2
    for row in range(len(class_labels)):
        for column in range(len(class_labels)):
            axis_matrix.text(
                column,
                row,
                str(confusion_matrix[row, column]),
                ha="center",
                va="center",
                color="white" if confusion_matrix[row, column] > threshold else "black",
                fontsize=12,
            )
    fig.colorbar(image, ax=axis_matrix, fraction=0.046, pad=0.04)

    order = np.argsort(importances)
    axis_importance.barh(
        np.asarray(FEATURE_NAMES)[order], importances[order], color="#2a9d8f"
    )
    axis_importance.set_title("Gini feature importance")
    axis_importance.set_xlabel("Normalized importance")
    axis_importance.set_xlim(0, max(0.5, float(importances.max()) * 1.15))

    fig.suptitle("From-scratch Random Forest — UCI Red Wine Quality")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if not args.data.exists():
        print(f"Dataset not found; downloading it from UCI to {args.data} ...")
        download_wine_data(args.data)

    X, y = load_wine_data(args.data)
    X_train, X_test, y_train, y_test = stratified_train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed
    )

    model = RandomForestClassifier(
        n_estimators=args.trees,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        max_features="sqrt",
        bootstrap=True,
        oob_score=True,
        random_state=args.seed,
    )
    started_at = time.perf_counter()
    model.fit(X_train, y_train)
    training_seconds = time.perf_counter() - started_at

    predictions = model.predict(X_test)
    scores, matrix, class_labels = multiclass_classification_metrics(
        y_test, predictions, labels=model.classes_
    )
    train_accuracy = model.score(X_train, y_train)
    majority_baseline = float(np.max(np.bincount(y_test)) / len(y_test))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "implementation": "from scratch using NumPy (no scikit-learn model)",
        "dataset": "UCI Red Wine Quality, dataset ID 186",
        "seed": args.seed,
        "n_estimators": args.trees,
        "max_depth": args.max_depth,
        "min_samples_leaf": args.min_samples_leaf,
        "source_samples": SOURCE_ROW_COUNT,
        "exact_duplicates_removed_before_split": DUPLICATE_ROWS_REMOVED,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "training_seconds": round(training_seconds, 4),
        "train_accuracy": train_accuracy,
        "oob_accuracy": model.oob_score_,
        "oob_coverage": model.oob_coverage_,
        "majority_class_baseline": majority_baseline,
        "test_metrics": scores,
        "class_labels": class_labels.tolist(),
        "confusion_matrix": matrix.tolist(),
        "feature_importance": dict(zip(FEATURE_NAMES, model.feature_importances_)),
        "target": "sensory quality score (observed classes 3 through 8)",
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "model.pkl").open("wb") as model_file:
        pickle.dump(model, model_file)
    create_evaluation_plot(
        matrix,
        class_labels,
        model.feature_importances_,
        args.output_dir / "evaluation.png",
    )

    print("\nTraining complete")
    print(f"  train/test rows : {len(X_train)} / {len(X_test)}")
    print(f"  trees           : {args.trees}")
    print(f"  max depth       : {args.max_depth}")
    print(f"  training time   : {training_seconds:.2f} seconds")
    print(f"  train accuracy  : {train_accuracy:.3%}")
    print(f"  OOB accuracy    : {model.oob_score_:.3%}")
    print(f"  test accuracy   : {scores['accuracy']:.3%}")
    print(f"  macro F1        : {scores['macro_f1']:.3%}")
    print(f"  within ±1 score : {scores['within_one_accuracy']:.3%}")
    print(f"  majority base   : {majority_baseline:.3%}")
    print(f"  outputs         : {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
