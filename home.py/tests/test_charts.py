"""Tests for the shared chart-building core (Phase 3)."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from utils import charts


@pytest.fixture
def df():
    rng = np.random.RandomState(0)
    return pd.DataFrame({
        "cat": rng.choice(["A", "B", "C"], 60),
        "cat2": rng.choice(["X", "Y"], 60),
        "num": rng.normal(10, 3, 60),
        "num2": rng.normal(5, 2, 60),
    })


def test_build_bar(df):
    assert isinstance(charts.build_bar(df, {"x": "cat", "y": "num", "aggregation": "mean"}), go.Figure)


def test_build_line(df):
    assert isinstance(charts.build_line(df, {"x": "cat", "y": "num", "aggregation": "sum"}), go.Figure)


def test_build_scatter(df):
    assert isinstance(charts.build_scatter(df, {"x": "num", "y": "num2"}), go.Figure)


def test_build_pie(df):
    assert isinstance(charts.build_pie(df, {"values": "num", "names": "cat"}), go.Figure)


def test_build_histogram(df):
    assert isinstance(charts.build_histogram(df, {"x": "num", "bins": 10}), go.Figure)


def test_build_box_and_violin(df):
    assert isinstance(charts.build_box(df, {"y": "num", "x": "cat"}), go.Figure)
    assert isinstance(charts.build_violin(df, {"y": "num", "x": "cat"}), go.Figure)


def test_build_kde_adds_real_curve(df):
    """The KDE fix overlays an actual Gaussian-KDE line (the old one didn't)."""
    fig = charts.build_kde(df, {"x": "num"})
    assert isinstance(fig, go.Figure)
    assert any(getattr(tr, "name", None) == "KDE" for tr in fig.data)


def test_build_heatmap(df):
    fig = charts.build_heatmap(df, {"x": "cat", "y": "cat2", "values": "num", "aggregation": "mean"})
    assert isinstance(fig, go.Figure)


def test_aggregate_none_returns_original(df):
    assert len(charts._aggregate(df, "cat", "num", "none")) == len(df)
