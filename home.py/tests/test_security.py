"""Security & robustness regression tests for Phase 2."""
import numpy as np
import pandas as pd
import pytest


def test_decorator_marks_transforms():
    """@record_step marks real transforms so replay can allowlist them."""
    from utils.numeric_processor import NumericProcessor
    proc = NumericProcessor()
    marked = [m for m in dir(proc)
              if not m.startswith("_")
              and getattr(getattr(proc, m), "_is_preprocessing_step", False)]
    assert marked  # at least one transform is marked as a preprocessing step


def test_pipeline_replay_rejects_non_step_function():
    """A crafted/imported pipeline JSON cannot invoke arbitrary methods."""
    from utils.preprocessing_pipeline import PreprocessingStep
    from utils.numeric_processor import NumericProcessor
    proc = NumericProcessor()
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    non_step = next(
        (m for m in dir(proc)
         if callable(getattr(proc, m)) and not m.startswith("_")
         and not getattr(getattr(proc, m), "_is_preprocessing_step", False)),
        "__class__",
    )
    step = PreprocessingStep("evil", "numeric", non_step, {"column": "x"})
    with pytest.raises(ValueError):
        step.apply(df, {"numeric": proc})


def test_chart_export_sanitizes_injection_in_name():
    """A column-derived chart name cannot inject script into the export HTML."""
    from utils.chart_export import _add_screenshot_capability
    malicious = "'); alert(document.cookie); //"
    out = _add_screenshot_capability("<html><body></body></html>", malicious)
    assert "alert(document.cookie)" not in out
    assert "'); alert" not in out


def test_data_quality_handles_empty_dataframe():
    from utils.data_quality import enhanced_data_quality
    report = enhanced_data_quality(pd.DataFrame({"a": [], "b": []}))
    assert isinstance(report, dict)
    assert any(r.get("issue") == "empty_dataset" for r in report["recommendations"])


def test_data_quality_handles_all_nan_column():
    """All-NaN numeric column (count == 0 / empty value_counts) must not crash."""
    from utils.data_quality import enhanced_data_quality
    df = pd.DataFrame({"all_nan": [np.nan, np.nan, np.nan], "ok": [1, 2, 3]})
    report = enhanced_data_quality(df)
    assert isinstance(report, dict)
