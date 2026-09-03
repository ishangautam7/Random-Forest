# Random Forest from Scratch: Red Wine Quality

A complete, beginner-friendly machine-learning project that predicts a red
wine's sensory quality score from 11 physicochemical measurements. The Random
Forest and its decision trees are implemented in this repository using
**NumPy only**—the project does not use a scikit-learn model.

This is a realistic multiclass problem. Unlike an unusually easy benchmark,
the classes overlap, quality judgments contain uncertainty, and the class
distribution is imbalanced. A moderate score is expected and is more useful for
discussing generalization and overfitting.

## What the project demonstrates

- CART decision trees with Gini-impurity splits
- Random feature selection at every tree node
- Bootstrap sampling and majority voting
- Out-of-bag (OOB) validation
- Explicit regularization using maximum depth and minimum leaf size
- Multiclass confusion matrix and macro-averaged metrics
- Accuracy within one quality point, useful for this ordered target
- Gini feature importance
- A saved model, JSON results, and evaluation chart

## Dataset

The project uses the red-wine portion of UCI's **Wine Quality** dataset. Its
source file has 1,599 Portuguese Vinho Verde rows, 11 continuous input features,
no missing values, and sensory quality ratings. The observed classes are 3–8.
The pipeline removes 240 exact duplicate records before splitting, leaving
1,359 unique modeling rows and preventing copies from leaking into the test set.

The input measurements are fixed acidity, volatile acidity, citric acid,
residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density,
pH, sulphates, and alcohol.

Source: [UCI Wine Quality, dataset 186](https://archive.ics.uci.edu/dataset/186/wine+quality)

See [DATASET_LICENSE.md](DATASET_LICENSE.md) for attribution and licensing.

## Run it

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py
```

The verified data file is included. If it is absent, training downloads the
official UCI file and verifies its SHA-256 checksum. It can also be downloaded
explicitly:

```bash
python scripts/download_data.py
```

Training creates:

- `artifacts/metrics.json` — train, OOB, test and baseline scores
- `artifacts/evaluation.png` — multiclass confusion matrix and feature importance
- `artifacts/model.pkl` — the trained Random Forest

Classify one wine after training. The script labels each of the 11 measurements
and shows its dataset range:

```bash
python predict.py
```

## Why this version is less overfit

Before fitting, the pipeline removes exact duplicate rows. The default forest
then limits every tree to depth 4 and requires at least five rows in each leaf.
These constraints prevent trees from simply memorizing individual training
rows. The project reports all three relevant measurements:

- **Training accuracy** reveals how closely the model fits known rows.
- **OOB accuracy** estimates generalization using training rows omitted from
  individual bootstrap samples.
- **Test accuracy** uses a separate stratified 20% holdout.

The gap between training and test accuracy matters more than whether the raw
accuracy seems high or low. You can explore the tradeoff explicitly:

```bash
# More regularized: likely lower training and test accuracy
python train.py --max-depth 3 --min-samples-leaf 10

# More flexible: likely higher training accuracy and a larger overfitting gap
python train.py --max-depth 10 --min-samples-leaf 2
```

## How the algorithm works

1. Each tree receives a bootstrap sample of the training rows.
2. At each node, it randomly selects `sqrt(number of features)` features.
3. It chooses the threshold with the largest decrease in weighted Gini impurity.
4. Depth and leaf-size constraints stop the tree from becoming too specific.
5. Each tree votes for a quality class; the most votes becomes the prediction.
6. Rows omitted from bootstrap samples provide the OOB evaluation.

The learning algorithm is in
[`random_forest_scratch/tree.py`](random_forest_scratch/tree.py) and
[`random_forest_scratch/forest.py`](random_forest_scratch/forest.py).

## Project structure

```text
random-forest/
├── data/winequality-red.csv      # verified UCI dataset
├── random_forest_scratch/
│   ├── data.py                   # loading, download, stratified split
│   ├── forest.py                 # bootstrap ensemble and voting
│   ├── metrics.py                # multiclass evaluation
│   └── tree.py                   # CART decision tree
├── scripts/download_data.py
├── train.py                      # training/evaluation pipeline
└── predict.py                    # one-row prediction demo
```

This is an educational implementation. Production ML libraries add substantial
speed, parallelism, serialization compatibility, and edge-case handling.
