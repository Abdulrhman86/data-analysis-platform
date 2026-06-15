"""Tests for the categorical preprocessing processor (Phase 5)."""
import pandas as pd

from utils.categorical_processor import CategoricalProcessor


def test_one_hot_encoding_drops_original_by_default():
    proc = CategoricalProcessor()
    df = pd.DataFrame({"color": ["red", "blue", "red"], "n": [1, 2, 3]})
    out = proc.apply_one_hot_encoding(df, "color")
    assert "color" not in out.columns
    assert {"color_red", "color_blue"}.issubset(out.columns)
    assert len(out) == 3


def test_one_hot_encoding_can_keep_original():
    proc = CategoricalProcessor()
    out = proc.apply_one_hot_encoding(pd.DataFrame({"color": ["red", "blue"]}),
                                      "color", keep_original=True)
    assert "color" in out.columns


def test_label_encoding_uses_sorted_mapping():
    proc = CategoricalProcessor()
    out = proc.apply_label_encoding(pd.DataFrame({"size": ["s", "m", "l", "s"]}), "size")
    assert "size_encoded" in out.columns
    assert out["size_encoded"].tolist() == [2, 1, 0, 2]  # sorted l=0, m=1, s=2


def test_merge_categories():
    proc = CategoricalProcessor()
    out = proc.merge_categories(pd.DataFrame({"c": ["a", "b", "c", "d"]}),
                                "c", ["a", "b"], "ab")
    assert out["c_merged"].tolist() == ["ab", "ab", "c", "d"]
