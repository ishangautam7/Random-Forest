from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

from random_forest_scratch.data import FEATURE_NAMES

FEATURE_RANGES = (
    "4.6-15.9",
    "0.12-1.58",
    "0-1",
    "0.9-15.5",
    "0.012-0.611",
    "1-72",
    "6-289",
    "0.99007-1.00369",
    "2.74-4.01",
    "0.33-2",
    "8.4-14.9",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict one red wine's quality.")
    parser.add_argument("--model", type=Path, default=Path("artifacts/model.pkl"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model.exists():
        raise SystemExit(f"Model not found at {args.model}. Run `python train.py` first.")

    with args.model.open("rb") as model_file:
        model = pickle.load(model_file)

    try:
        features = np.array(
            [[
                float(input(f"{label} ({value_range}): "))
                for label, value_range in zip(FEATURE_NAMES, FEATURE_RANGES)
            ]],
            dtype=float,
        )
    except (EOFError, ValueError):
        raise SystemExit("Enter a valid number for every feature.") from None
    predicted_class = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]

    print(f"Predicted wine quality: {predicted_class}/10")
    vote_text = ", ".join(
        f"quality {label}={probability:.1%}"
        for label, probability in zip(model.classes_, probabilities)
    )
    print(f"Tree votes: {vote_text}")


if __name__ == "__main__":
    main()
