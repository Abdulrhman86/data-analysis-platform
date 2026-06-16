"""Phase 6 — feature engineering transforms (previously untested)."""
import pandas as pd

from utils.feature_engineering import FeatureEngineeringProcessor


def test_polynomial_features():
    fe = FeatureEngineeringProcessor()
    out = fe.create_polynomial_features(pd.DataFrame({"x": [2.0, 3.0]}), "x", degree=3)
    assert out["x^2"].tolist() == [4.0, 9.0]
    assert out["x^3"].tolist() == [8.0, 27.0]


def test_interaction_multiply():
    fe = FeatureEngineeringProcessor()
    out = fe.create_interaction_term(
        pd.DataFrame({"a": [2, 3], "b": [4, 5]}), "a", "b", operation="multiply")
    assert out["a_x_b"].tolist() == [8, 15]


def test_interaction_divide_handles_zero():
    fe = FeatureEngineeringProcessor()
    out = fe.create_interaction_term(
        pd.DataFrame({"a": [4.0], "b": [0.0]}), "a", "b", operation="divide")
    assert pd.isna(out["a_div_b"].iloc[0])   # divide-by-zero -> NaN, not inf/crash


def test_binned_feature():
    fe = FeatureEngineeringProcessor()
    out = fe.create_binned_feature(pd.DataFrame({"x": range(20)}), "x", bins=4)
    assert "x_binned" in out.columns
    assert out["x_binned"].nunique() == 4


def test_aggregation_features_map_back():
    fe = FeatureEngineeringProcessor()
    df = pd.DataFrame({"g": ["a", "a", "b"], "v": [10.0, 20.0, 30.0]})
    out = fe.create_aggregation_features(df, "g", "v", agg_functions=["mean"])
    assert out["v_mean_by_g"].tolist() == [15.0, 15.0, 30.0]
