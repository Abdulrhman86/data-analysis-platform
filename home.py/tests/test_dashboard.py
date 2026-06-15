"""Tests for dashboard persistence and serialization (Phase 3)."""
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
