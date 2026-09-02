"""Load the trained model and predict one red wine's quality score."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict one red wine's quality.")
    parser.add_argument("fixed_acidity", type=float)
    parser.add_argument("volatile_acidity", type=float)
    parser.add_argument("citric_acid", type=float)
    parser.add_argument("residual_sugar", type=float)
    parser.add_argument("chlorides", type=float)
    parser.add_argument("free_sulfur_dioxide", type=float)
    parser.add_argument("total_sulfur_dioxide", type=float)
    parser.add_argument("density", type=float)
    parser.add_argument("ph", type=float)
    parser.add_argument("sulphates", type=float)
    parser.add_argument("alcohol", type=float)
    parser.add_argument("--model", type=Path, default=Path("artifacts/model.pkl"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model.exists():
        raise SystemExit(f"Model not found at {args.model}. Run `python train.py` first.")

    with args.model.open("rb") as model_file:
        model = pickle.load(model_file)

    features = np.array(
        [[
            args.fixed_acidity,
            args.volatile_acidity,
            args.citric_acid,
            args.residual_sugar,
            args.chlorides,
            args.free_sulfur_dioxide,
            args.total_sulfur_dioxide,
            args.density,
            args.ph,
            args.sulphates,
            args.alcohol,
        ]],
        dtype=float,
    )
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
