"""Phase 5 — dashboard persistence robustness.

A single malformed record (or a truncated file) used to make load_dashboards
return {} — silently wiping every saved dashboard. These verify fault isolation,
atomic writes, and quarantine-on-corruption.
"""
import json
import os

from utils.dashboard_module import (
    Dashboard, save_dashboards, load_dashboards, _dashboards_file,
)


def _use_tmp(monkeypatch, tmp_path):
    monkeypatch.setenv("DASHBOARD_DIR", str(tmp_path))


def test_save_load_round_trip(monkeypatch, tmp_path):
    _use_tmp(monkeypatch, tmp_path)
    save_dashboards({"D1": Dashboard("D1"), "D2": Dashboard("D2")})
    assert set(load_dashboards().keys()) == {"D1", "D2"}


def test_missing_file_returns_empty(monkeypatch, tmp_path):
    _use_tmp(monkeypatch, tmp_path)
    assert load_dashboards() == {}


def test_atomic_write_leaves_no_tmp(monkeypatch, tmp_path):
    _use_tmp(monkeypatch, tmp_path)
    save_dashboards({"D1": Dashboard("D1")})
    assert not os.path.exists(_dashboards_file() + ".tmp")


def test_corrupt_file_is_quarantined(monkeypatch, tmp_path):
    _use_tmp(monkeypatch, tmp_path)
    path = _dashboards_file()
    with open(path, "w") as f:
        f.write("{ this is not valid json")
    assert load_dashboards() == {}
    # quarantined to .corrupt rather than left to be overwritten + lost
    assert os.path.exists(path + ".corrupt")


def test_one_bad_record_does_not_kill_the_rest(monkeypatch, tmp_path):
    _use_tmp(monkeypatch, tmp_path)
    path = _dashboards_file()
    payload = {
        "good": Dashboard("good").to_dict(),
        "bad": {"description": "missing the required 'name' key"},  # from_dict -> KeyError
    }
    with open(path, "w") as f:
        json.dump(payload, f)
    loaded = load_dashboards()
    assert "good" in loaded
    assert "bad" not in loaded


def test_non_dict_json_returns_empty(monkeypatch, tmp_path):
    _use_tmp(monkeypatch, tmp_path)
    with open(_dashboards_file(), "w") as f:
        json.dump([1, 2, 3], f)
    assert load_dashboards() == {}
