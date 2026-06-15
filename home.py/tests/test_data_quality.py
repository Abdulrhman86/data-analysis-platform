"""Tests for the data-quality engine (Phase 5)."""
import pandas as pd

from utils.data_quality import enhanced_data_quality


def _issues(report):
    return {r["issue"] for r in report["recommendations"]}


def test_summary_counts():
    report = enhanced_data_quality(pd.DataFrame({"a": [1, 2, 2], "b": ["x", "y", "y"]}))
    assert report["summary"]["row_count"] == 3
    assert report["summary"]["column_count"] == 2


def test_flags_constant_column():
    df = pd.DataFrame({"const": [5, 5, 5, 5], "x": [1, 2, 3, 4]})
    assert "constant" in _issues(enhanced_data_quality(df))


def test_flags_outliers():
    df = pd.DataFrame({"v": [1, 2, 3, 4, 5, 6, 7, 8, 9, 1000]})
    assert "outliers" in _issues(enhanced_data_quality(df))


def test_flags_high_cardinality():
    df = pd.DataFrame({"id": [f"cat_{i}" for i in range(120)]})
    assert "high_cardinality" in _issues(enhanced_data_quality(df))


def test_missing_values_produce_recommendation():
    df = pd.DataFrame({"a": [1, None, None, None, 5]})
    assert enhanced_data_quality(df)["recommendations"]
