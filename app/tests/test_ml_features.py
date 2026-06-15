"""Tests for Phase 4 ML enhancements: new models, tuning, configurable scaler."""
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import RobustScaler

from utils.classification_models import ClassificationProcessor
from utils.regression_models import RegressionProcessor


@pytest.fixture
def clf_data():
    rng = np.random.RandomState(0)
    n = 200
    df = pd.DataFrame({
        "age": rng.randint(18, 70, n),
        "income": rng.normal(50000, 15000, n),
        "city": rng.choice(["NY", "LA", "SF"], n),
    })
    df["label"] = (df["income"] > 50000).astype(int)
    clf = ClassificationProcessor()
    split = clf.prepare_data(df, "label", ["age", "income", "city"], test_size=0.25, random_state=0)
    return clf, split


def test_new_classification_models_registered(clf_data):
    clf, _ = clf_data
    assert {"K-Nearest Neighbors", "Gradient Boosting"}.issubset(clf.models)


def test_new_models_train_and_evaluate(clf_data):
    clf, s = clf_data
    for mdl in ["K-Nearest Neighbors", "Gradient Boosting"]:
        clf.train_model(mdl, s["X_train"], s["y_train"])
        assert "accuracy" in clf.evaluate_model(mdl, s["X_test"], s["y_test"])


def test_tune_model_returns_best_params(clf_data):
    clf, s = clf_data
    _, best_params = clf.tune_model("K-Nearest Neighbors", s["X_train"], s["y_train"], cv=3)
    assert "n_neighbors" in best_params
    assert "accuracy" in clf.evaluate_model("K-Nearest Neighbors", s["X_test"], s["y_test"])


def test_configurable_scaler(clf_data):
    clf, s = clf_data
    clf.preprocessing["scaler"] = "robust"
    clf.train_model("Logistic Regression", s["X_train"], s["y_train"])
    pre = clf.get_fitted("Logistic Regression").named_steps["preprocess"]
    scaler = pre.named_transformers_["num"].named_steps["scale"]
    assert isinstance(scaler, RobustScaler)


def test_new_regression_models():
    rng = np.random.RandomState(1)
    n = 150
    df = pd.DataFrame({"x1": rng.normal(0, 1, n), "x2": rng.normal(5, 2, n)})
    df["y"] = 3 * df["x1"] - 2 * df["x2"] + rng.normal(0, 0.5, n)
    reg = RegressionProcessor()
    assert {"Elastic Net", "K-Nearest Neighbors", "Gradient Boosting"}.issubset(reg.models)
    s = reg.prepare_data(df, "y", ["x1", "x2"], test_size=0.25, random_state=0)
    for mdl in ["Elastic Net", "K-Nearest Neighbors", "Gradient Boosting"]:
        reg.train_model(mdl, s["X_train"], s["y_train"])
        assert "r2" in reg.evaluate_model(mdl, s["X_test"], s["y_test"])


def test_predict_on_new_dataframe(clf_data):
    """The 'Predict' tab scores new raw data via the fitted pipeline."""
    clf, s = clf_data
    clf.train_model("Gradient Boosting", s["X_train"], s["y_train"])
    model = clf.get_fitted("Gradient Boosting")
    new = pd.DataFrame({"age": [25, 60], "income": [40000.0, 70000.0], "city": ["NY", "LA"]})
    preds = model.predict(new[["age", "income", "city"]])
    assert len(preds) == 2


def test_class_weight_balanced_default():
    clf = ClassificationProcessor()
    for name in ["Logistic Regression", "Decision Tree", "Random Forest", "SVM", "Gradient Boosting"]:
        assert clf.models[name].get_params().get("class_weight") == "balanced"
    # KNN and Naive Bayes have no class_weight param and are left untouched
    assert "class_weight" not in clf.models["K-Nearest Neighbors"].get_params()
    assert "class_weight" not in clf.models["Naive Bayes"].get_params()
