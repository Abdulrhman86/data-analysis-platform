"""Regression tests for the Phase 1 ML pipeline.

These lock in the fixes that wrap every model in an sklearn
Pipeline + ColumnTransformer (impute -> encode categoricals -> scale numerics)
fit only on the training split.
"""
import json
import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from utils.classification_models import ClassificationProcessor
from utils.regression_models import RegressionProcessor
from utils.model_evaluation import get_precision_recall_curve_data


@pytest.fixture
def mixed_df():
    """Numeric + string-categorical features, with missing values in features."""
    rng = np.random.RandomState(0)
    n = 240
    df = pd.DataFrame({
        "age": rng.randint(18, 70, n),
        "income": rng.normal(50000, 15000, n),
        "city": rng.choice(["NY", "LA", "SF", "TX"], n),
        "tier": rng.choice(["A", "B", "C"], n),
    })
    df["segment"] = np.where(df["income"] > 55000, "high",
                             np.where(df["income"] > 45000, "mid", "low"))
    df.loc[df.sample(25, random_state=1).index, "income"] = np.nan
    df.loc[df.sample(18, random_state=2).index, "city"] = np.nan
    return df


def _clf_split(df):
    clf = ClassificationProcessor()
    split = clf.prepare_data(df, "segment", ["age", "income", "city", "tier"],
                             test_size=0.25, random_state=42)
    return clf, split


def test_classification_trains_on_mixed_types(mixed_df):
    """The headline fix: string features no longer crash training."""
    clf, s = _clf_split(mixed_df)
    for mdl in ["Logistic Regression", "Random Forest", "SVM", "Naive Bayes"]:
        clf.train_model(mdl, s["X_train"], s["y_train"])
        metrics = clf.evaluate_model(mdl, s["X_test"], s["y_test"])
        assert "accuracy" in metrics
        assert metrics["roc_auc"] is None or 0.0 <= metrics["roc_auc"] <= 1.0


def test_split_keeps_categoricals_and_stratifies(mixed_df):
    _, s = _clf_split(mixed_df)
    assert {"city", "tier"}.issubset(s["X_train"].columns)
    assert s["y_test"].nunique() == 3  # stratified -> every class present in test


def test_numeric_features_are_scaled(mixed_df):
    clf, s = _clf_split(mixed_df)
    clf.train_model("Logistic Regression", s["X_train"], s["y_train"])
    pre = clf.get_fitted("Logistic Regression").named_steps["preprocess"]
    transformed = pre.transform(s["X_train"])
    assert abs(transformed[:, 0].mean()) < 0.2  # age standardized
    assert abs(transformed[:, 1].mean()) < 0.2  # income standardized


def test_feature_importance_uses_encoded_names(mixed_df):
    clf, s = _clf_split(mixed_df)
    clf.train_model("Random Forest", s["X_train"], s["y_train"])
    importance = clf.get_feature_importance("Random Forest")
    assert any("city" in str(idx) for idx in importance.index)


def test_cross_validate_on_mixed_data(mixed_df):
    clf, s = _clf_split(mixed_df)
    scores = clf.cross_validate("Logistic Regression", s["X_train"], s["y_train"],
                                cv=3, scoring="accuracy")
    assert len(scores) == 3


def test_exported_pipeline_predicts_raw_data(mixed_df):
    """Exported model carries its preprocessing -> usable on raw, unseen data."""
    clf, s = _clf_split(mixed_df)
    clf.train_model("Random Forest", s["X_train"], s["y_train"])
    reloaded = pickle.loads(pickle.dumps(clf.get_fitted("Random Forest")))
    new_raw = pd.DataFrame({"age": [30, 55], "income": [40000.0, 60000.0],
                            "city": ["NY", "ZZ"], "tier": ["A", "C"]})  # 'ZZ' unseen
    assert len(reloaded.predict(new_raw)) == 2


def test_update_params_then_retrain(mixed_df):
    clf, s = _clf_split(mixed_df)
    clf.update_model_parameters("Random Forest",
                                {"n_estimators": 50, "max_depth": 4,
                                 "min_samples_split": 2, "min_samples_leaf": 1})
    clf.train_model("Random Forest", s["X_train"], s["y_train"])
    assert "Random Forest" in clf.fitted_models


def test_regression_trains_and_rejects_non_numeric_target(mixed_df):
    reg = RegressionProcessor()
    r = reg.prepare_data(mixed_df, "income", ["age"], test_size=0.25, random_state=42)
    for mdl in ["Linear Regression", "Ridge Regression", "Random Forest Regressor"]:
        reg.train_model(mdl, r["X_train"], r["y_train"])
        assert "r2" in reg.evaluate_model(mdl, r["X_test"], r["y_test"])
    with pytest.raises(ValueError):
        reg.prepare_data(mixed_df, "city", ["age"])  # non-numeric target


def test_average_precision_matches_sklearn(mixed_df):
    clf = ClassificationProcessor()
    df = mixed_df.copy()
    df["is_high"] = (df["segment"] == "high").astype(int)
    s = clf.prepare_data(df, "is_high", ["age", "income", "city", "tier"],
                         test_size=0.3, random_state=0)
    clf.train_model("Logistic Regression", s["X_train"], s["y_train"])
    fitted = clf.get_fitted("Logistic Regression")
    pr = get_precision_recall_curve_data(fitted, s["X_test"], s["y_test"])
    proba = fitted.predict_proba(s["X_test"])[:, 1]
    assert abs(pr["avg_precision"] - average_precision_score(s["y_test"], proba)) < 1e-9


def test_metadata_with_confusion_matrix_is_json_safe(mixed_df):
    from utils.model_export import _json_default
    clf, s = _clf_split(mixed_df)
    clf.train_model("Random Forest", s["X_train"], s["y_train"])
    meta = {"metrics": clf.evaluate_model("Random Forest", s["X_test"], s["y_test"])}
    json.dumps(meta, default=_json_default)  # numpy confusion_matrix must not raise
