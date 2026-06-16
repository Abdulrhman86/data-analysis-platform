"""Phase 4 — replay correctness.

Recorded preprocessing steps are now re-applied to new data (Predict). These
verify the transforms stay correct on replay: log of non-positive values is
finite, and strategy-based imputation recomputes the statistic from the new
data instead of using a value frozen when the recipe was recorded.
"""
import numpy as np
import pandas as pd

from utils.numeric_processor import NumericProcessor
from utils.data_processor import DataProcessor


def test_log_transform_is_finite_on_nonpositive():
    out = NumericProcessor().apply_log_transform(
        pd.DataFrame({"x": [0.0, -5.0, 10.0]}), "x")
    # zeros would give -inf and negatives NaN without the clip guard
    assert np.isfinite(out["x_log"]).all()


def test_log_transform_preserves_nan():
    out = NumericProcessor().apply_log_transform(
        pd.DataFrame({"x": [1.0, np.nan, 100.0]}), "x")
    assert out["x_log"].isna().sum() == 1


def test_imputation_strategy_recomputes_on_new_data():
    dp = DataProcessor()
    out1 = dp.replace_missing_values(pd.DataFrame({"x": [1.0, 3.0, np.nan]}),
                                     "x", strategy="mean")
    out2 = dp.replace_missing_values(pd.DataFrame({"x": [10.0, 30.0, np.nan]}),
                                     "x", strategy="mean")
    assert out1["x"].iloc[2] == 2.0    # mean of [1, 3]
    assert out2["x"].iloc[2] == 20.0   # mean of [10, 30] -> recomputed, not frozen at 2.0


def test_imputation_literal_value_still_works():
    dp = DataProcessor()
    out = dp.replace_missing_values(pd.DataFrame({"x": [1.0, np.nan]}), "x", value=99.0)
    assert out["x"].iloc[1] == 99.0


def test_imputation_most_frequent_recomputes():
    dp = DataProcessor()
    out = dp.replace_missing_values(pd.DataFrame({"c": ["a", "a", "b", None]}),
                                    "c", strategy="most_frequent")
    assert out["c"].iloc[3] == "a"
