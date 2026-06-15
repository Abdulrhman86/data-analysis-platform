"""End-to-end workflow smoke test on the shipped sample dataset.

Mirrors the manual browser walkthrough (Phase A) in code: load -> data quality ->
datetime feature extraction -> train a classifier on data WITH categorical
features (the Phase 1 fix) -> predict on new rows via the fitted pipeline
(Phase 4b). Catches integration regressions across the workflow without a browser.
"""
import os

import pandas as pd

from utils.data_quality import enhanced_data_quality
from utils.datetime_processor import DatetimeProcessor
from utils.classification_models import ClassificationProcessor

SAMPLE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static", "sample_sales.csv")

# Numeric + categorical mix; excludes the high-cardinality id for speed.
FEATURES = ["region", "product_category", "units", "unit_price", "discount",
            "satisfaction", "revenue", "order_date_year", "order_date_month",
            "order_date_quarter"]


def test_full_workflow_on_sample_dataset():
    # 1. Upload / load the shipped messy sample
    df = pd.read_csv(SAMPLE, parse_dates=["order_date"])
    assert len(df) == 220
    assert df.isna().sum().sum() > 0  # deliberately messy

    # 2. Data-quality assessment runs and flags issues
    report = enhanced_data_quality(df)
    assert report["summary"]["row_count"] == 220
    assert report["recommendations"]

    # 3. Preprocessing: extract datetime components (as the app does)
    df2 = DatetimeProcessor().extract_components(df, "order_date")
    assert {"order_date_year", "order_date_quarter"}.issubset(df2.columns)

    # 4. Train a classifier on data WITH string/categorical features (Phase 1)
    clf = ClassificationProcessor()
    split = clf.prepare_data(df2, "churned", FEATURES, test_size=0.2, random_state=42)
    clf.train_model("Logistic Regression", split["X_train"], split["y_train"])
    metrics = clf.evaluate_model("Logistic Regression", split["X_test"], split["y_test"])
    assert "accuracy" in metrics

    # 5. Predict on new raw rows via the fitted pipeline (Phase 4b)
    preds = clf.get_fitted("Logistic Regression").predict(df2[FEATURES].head(10))
    assert len(preds) == 10
