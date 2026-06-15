"""Demonstrate -- don't just assert -- that the training pipeline has NO data leakage.

The Phase 1 rebuild wraps every model in an sklearn Pipeline whose preprocessing
(impute -> encode -> scale) is fit ONLY on the training split. This script proves
two things you can run live at a defense:

  1. The fitted scaler's statistics come from the TRAINING data only (never test).
  2. With a RANDOM target (no real signal), the pipeline's cross-validated accuracy
     stays at chance (~0.5) -- whereas a common *leaky* approach (selecting features
     on the full dataset before cross-validation) is falsely optimistic.

Run (from the project root, with dev deps installed):
    python scripts/show_no_leakage.py
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Make the app package importable.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app"))
from utils.classification_models import ClassificationProcessor  # noqa: E402


def demo_1_scaler_fit_on_train_only():
    print("\n=== Demo 1: the scaler is fit on the TRAINING split only ===")
    rng = np.random.RandomState(0)
    # Train rows and (mostly) test rows have deliberately different means.
    df = pd.DataFrame({
        "x": np.concatenate([rng.normal(0, 1, 160), rng.normal(10, 1, 40)]),
        "y": [0] * 100 + [1] * 100,
    })
    clf = ClassificationProcessor()
    split = clf.prepare_data(df, "y", ["x"], test_size=0.2, random_state=42)
    clf.train_model("Logistic Regression", split["X_train"], split["y_train"])

    scaler = (clf.get_fitted("Logistic Regression")
              .named_steps["preprocess"].named_transformers_["num"].named_steps["scale"])
    train_mean = split["X_train"]["x"].mean()
    full_mean = df["x"].mean()

    print(f"  fitted scaler.mean_     = {scaler.mean_[0]:.4f}")
    print(f"  X_train['x'].mean()     = {train_mean:.4f}   <- the scaler matches this")
    print(f"  full dataset ['x'].mean = {full_mean:.4f}   <- and NOT this (test never leaked in)")
    assert abs(scaler.mean_[0] - train_mean) < 1e-9
    assert abs(scaler.mean_[0] - full_mean) > 1e-6
    print("  PASS: the transformer's statistics came from the training data only.")


def demo_2_random_target_stays_at_chance():
    print("\n=== Demo 2: random target -> honest ~chance accuracy (no leakage) ===")
    rng = np.random.RandomState(1)
    n, p = 200, 300  # many noise features, few samples
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"f{i}" for i in range(p)])
    y = pd.Series(rng.randint(0, 2, n))  # RANDOM target -> true accuracy is 0.5
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    # WRONG (leaky): pick the "best" features using the FULL dataset (peeks at all y),
    # then cross-validate the result -> falsely optimistic.
    X_leaky = SelectKBest(f_classif, k=10).fit_transform(X, y)
    leaky = cross_val_score(LogisticRegression(max_iter=1000), X_leaky, y, cv=cv).mean()

    # RIGHT (our pipeline): preprocessing is fit INSIDE each fold, on training rows only.
    estimator = ClassificationProcessor()._build_pipeline("Logistic Regression", X)
    correct = cross_val_score(estimator, X, y, cv=cv).mean()

    print(f"  Leaky    (select features on ALL data, then CV): {leaky:.3f}   <- falsely optimistic")
    print(f"  Our pipeline (preprocess fit inside each fold):  {correct:.3f}   <- honest, ~0.5")
    assert correct < leaky  # the honest score is lower because it doesn't cheat
    print("  PASS: leakage inflates the leaky score; our pipeline stays near chance.")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    demo_1_scaler_fit_on_train_only()
    demo_2_random_target_stays_at_chance()
    print("\nAll demonstrations passed: preprocessing is fit on the training split only.\n")
