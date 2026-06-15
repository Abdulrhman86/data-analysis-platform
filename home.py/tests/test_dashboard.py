"""Tests for dashboard persistence, serialization, filters, and HTML export."""
import pandas as pd

from utils import dashboard_module as dm


def test_component_serialization_round_trip():
    comp = dm.DashboardComponent(title="Bar", chart_type="bar",
                                 config={"x": "a", "y": "b", "aggregation": "sum"},
                                 width=6, height=400)
    restored = dm.DashboardComponent.from_dict(comp.to_dict())
    assert restored.chart_type == "bar"
    assert restored.config == {"x": "a", "y": "b", "aggregation": "sum"}
    assert restored.width == 6


def test_dashboard_persistence_round_trip(tmp_path, monkeypatch):
    target = tmp_path / "dashboards.json"
    monkeypatch.setattr(dm, "_dashboards_file", lambda: str(target))

    dash = dm.Dashboard("My Dash", "a description")
    dash.add_component(dm.DashboardComponent(title="Scatter", chart_type="scatter",
                                             config={"x": "a", "y": "b"}))
    dm.save_dashboards({"My Dash": dash})

    assert target.exists()
    loaded = dm.load_dashboards()
    assert "My Dash" in loaded
    assert loaded["My Dash"].description == "a description"
    assert len(loaded["My Dash"].components) == 1
    assert loaded["My Dash"].components[0].chart_type == "scatter"


def test_load_dashboards_missing_file_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(dm, "_dashboards_file", lambda: str(tmp_path / "does_not_exist.json"))
    assert dm.load_dashboards() == {}


def test_get_figure_common_and_unknown():
    df = pd.DataFrame({"cat": ["a", "b", "a"], "num": [1.0, 2.0, 3.0]})
    assert dm.DashboardComponent("Bar", "bar", {"x": "cat", "y": "num"}).get_figure(df) is not None
    assert dm.DashboardComponent("Radar", "custom_radar", {}).get_figure(df) is None


def test_export_dashboard_html():
    df = pd.DataFrame({"cat": ["a", "b", "a", "b"], "num": [1.0, 2.0, 3.0, 4.0]})
    dash = dm.Dashboard("D")
    dash.add_component(dm.DashboardComponent("Bar", "bar", {"x": "cat", "y": "num", "aggregation": "sum"}))
    dash.add_component(dm.DashboardComponent("Scatter", "scatter", {"x": "num", "y": "num"}))
    out = dm.export_dashboard_html(dash, df)
    assert out is not None and "<html" in out.lower()
    assert "plotly" in out.lower()  # at least one chart embedded


def test_export_dashboard_html_no_exportable_returns_none():
    df = pd.DataFrame({"a": [1, 2, 3]})
    dash = dm.Dashboard("Empty")
    dash.add_component(dm.DashboardComponent("Radar", "custom_radar", {}))  # not exportable
    assert dm.export_dashboard_html(dash, df) is None
