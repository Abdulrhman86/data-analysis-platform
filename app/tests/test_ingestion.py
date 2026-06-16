"""Phase 1 — real-world data ingestion & type robustness.

Covers the parse core factored out of the Streamlit upload page so messy real
files (numbers-as-text, duplicate headers, odd delimiters, non-UTF-8, Excel
serial dates) load and type correctly instead of silently breaking.
"""
import io
import pandas as pd

from utils.preprocessing_utils import (
    coerce_numeric_strings,
    dedupe_column_names,
    detect_column_types,
    read_csv_robust,
    prepare_excel_datetime_columns,
    finalize_dataframe,
)


# ---------- coerce_numeric_strings ----------

def test_coerce_numeric_strings_converts_all_numeric_text():
    df = pd.DataFrame({"x": ["1", "2", "3"], "y": ["1.5", "2.0", "-3"]})
    out = coerce_numeric_strings(df)
    assert pd.api.types.is_numeric_dtype(out["x"])
    assert pd.api.types.is_numeric_dtype(out["y"])
    assert out["x"].tolist() == [1, 2, 3]


def test_coerce_numeric_strings_leaves_mixed_column():
    out = coerce_numeric_strings(pd.DataFrame({"c": ["1", "2", "apple"]}))
    assert out["c"].dtype == object  # a single non-number keeps the column as text


def test_coerce_numeric_strings_handles_na_markers():
    out = coerce_numeric_strings(pd.DataFrame({"x": ["1", None, "3"]}))
    assert pd.api.types.is_numeric_dtype(out["x"])


def test_coerce_numeric_strings_leaves_real_text():
    out = coerce_numeric_strings(pd.DataFrame({"city": ["NYC", "LA", "SF"]}))
    assert out["city"].dtype == object


# ---------- dedupe_column_names ----------

def test_dedupe_column_names_renames_duplicates():
    df = pd.DataFrame([[1, 2, 3]], columns=["a", "a", "b"])
    assert list(dedupe_column_names(df).columns) == ["a", "a.1", "b"]


def test_dedupe_column_names_noop_when_unique():
    df = pd.DataFrame({"a": [1], "b": [2]})
    assert list(dedupe_column_names(df).columns) == ["a", "b"]


# ---------- detect_column_types ----------

def test_detect_numeric_string_is_numeric_not_orphan():
    types = detect_column_types(pd.DataFrame({"x": ["1", "2", "3"]}))
    assert types["x"] == "numeric"  # was the orphan 'numeric_as_string'


def test_detect_basic_types():
    df = pd.DataFrame({
        "n": [1, 2, 3],
        "c": ["a", "b", "c"],
        "d": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"]),
    })
    types = detect_column_types(df)
    assert types["n"] == "numeric"
    assert types["c"] == "categorical"
    assert types["d"] == "datetime"


def test_detect_never_returns_orphan_label():
    types = detect_column_types(pd.DataFrame({"x": ["1", "2"], "y": ["a", "b"]}))
    assert "numeric_as_string" not in types.values()


# ---------- read_csv_robust ----------

def test_read_csv_comma():
    df = read_csv_robust(io.StringIO("a,b\n1,2\n3,4\n"), header=0)
    assert list(df.columns) == ["a", "b"]
    assert df.shape == (2, 2)


def test_read_csv_semicolon_sniffed():
    df = read_csv_robust(io.StringIO("a;b;c\n1;2;3\n4;5;6\n"), header=0)
    assert list(df.columns) == ["a", "b", "c"]  # delimiter sniffed
    assert df.shape == (2, 3)


def test_read_csv_latin1_fallback():
    raw = "name,city\n1,caf\xe9\n".encode("latin-1")  # invalid utf-8 -> must fall back
    df = read_csv_robust(io.BytesIO(raw), header=0)
    assert df.shape == (1, 2)


# ---------- prepare_excel_datetime_columns ----------

def test_excel_dates_only_convert_date_named_columns():
    df = pd.DataFrame({
        "order_date": [44000, 44100, 44200],  # serial dates + a date-ish name
        "price": [44000, 44100, 44200],        # same range, NOT a date name
    })
    out = prepare_excel_datetime_columns(df)
    assert pd.api.types.is_datetime64_any_dtype(out["order_date"])
    assert pd.api.types.is_numeric_dtype(out["price"])  # left alone


# ---------- finalize_dataframe (end to end) ----------

def test_finalize_dedupes_and_coerces():
    df = pd.DataFrame([["1", "x", "9"]], columns=["a", "b", "a"])
    out = finalize_dataframe(df)
    assert list(out.columns) == ["a", "b", "a.1"]
    assert pd.api.types.is_numeric_dtype(out["a"])
    assert pd.api.types.is_numeric_dtype(out["a.1"])
