"""Phase 3 — ML edge-case hardening.

Small/imbalanced data, constant and high-cardinality columns, and categorical
features in regression must no longer crash or silently misbehave.
"""
import numpy as np
import pandas as pd

from utils.classification_models import ClassificationProcessor
from utils.regression_models import RegressionProcessor


def _first_model(proc):
    return list(proc.models.keys())[0]


def test_safe_cv_folds_clamps_to_smallest_class():
    proc = ClassificationProcessor()
    assert proc.safe_cv_folds(pd.Series([0, 0, 0, 1, 1]), 5) == 2   # min class = 2
    assert proc.safe_cv_folds(pd.Series([0, 0, 0, 1]), 5) is None    # singleton class


def test_cross_validate_clamps_on_small_imbalanced_data():
    df = pd.DataFrame({
        "a": list(range(12)),
        "b": list(range(12, 24)),
        "target": [0] * 9 + [1] * 3,    # minority class smaller than the 5 folds asked for
    })
    proc = ClassificationProcessor()
    split = proc.prepare_data(df, "target", test_size=0.25)
    # cv=5 requested but clamped internally -> must not raise
    scores = proc.cross_validate(_first_model(proc), split["X_train"], split["y_train"],
                                 cv=5, scoring="accuracy")
    assert len(scores) >= 2


def test_constant_feature_does_not_break_training():
    rng = np.random.RandomState(0)
    df = pd.DataFrame({
        "a": rng.rand(40),
        "const": np.ones(40),            # zero-variance numeric column
        "target": rng.choice([0, 1], 40),
    })
    proc = ClassificationProcessor()
    split = proc.prepare_data(df, "target", test_size=0.3)
    model = proc.train_model(_first_model(proc), split["X_train"], split["y_train"])
    assert len(model.predict(split["X_test"])) == len(split["X_test"])


def test_regression_accepts_categorical_features():
    rng = np.random.RandomState(1)
    df = pd.DataFrame({
        "num": rng.rand(40),
        "cat": rng.choice(["p", "q"], 40),  # categorical predictor for regression
        "y": rng.rand(40),
    })
    proc = RegressionProcessor()
    split = proc.prepare_data(df, "y", test_size=0.3)
    model = proc.train_model(_first_model(proc), split["X_train"], split["y_train"])
    assert len(model.predict(split["X_test"])) == len(split["X_test"])


def test_high_cardinality_categorical_is_capped():
    rng = np.random.RandomState(2)
    df = pd.DataFrame({
        "id": [f"u{i}" for i in range(120)],   # 120 unique categories
        "n": rng.rand(120),
        "target": rng.choice([0, 1], 120),
    })
    proc = ClassificationProcessor()
    split = proc.prepare_data(df, "target", test_size=0.3)
    model = proc.train_model(_first_model(proc), split["X_train"], split["y_train"])
    width = model.named_steps["preprocess"].transform(split["X_train"]).shape[1]
    assert width <= 55   # max_categories caps the one-hot blow-up (not ~80+)
