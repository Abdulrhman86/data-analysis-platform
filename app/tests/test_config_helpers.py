"""Phase 6 — Config UI helpers (stepper / metric_card / empty_state)."""
from config import Config


def test_stepper_marks_done_current_upcoming():
    html = Config.stepper(3)
    assert html.count('class="step done"') == 2     # steps 1, 2 completed
    assert 'class="step current"' in html           # step 3 current
    assert "Prepare" in html                         # step 3 label


def test_stepper_first_step_has_no_done():
    html = Config.stepper(1)
    assert html.count('class="step done"') == 0
    assert 'class="step current"' in html


def test_stepper_last_step_all_prior_done():
    assert Config.stepper(6).count('class="step done"') == 5


def test_metric_card_wraps_label_and_value():
    html = Config.metric_card("Rows", "220")
    assert "metric-container" in html
    assert "Rows" in html and "220" in html


def test_empty_state_renders_title_and_message():
    html = Config.empty_state("No data", "Upload first", icon="📂")
    assert "empty-state" in html
    assert "No data" in html and "Upload first" in html
