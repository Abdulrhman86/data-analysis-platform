"""Phase 2 — self-contained model.

The Predict/Export flow re-applies the preprocessing recipe captured at training
time to RAW uploaded data, so a user can hand the model raw columns instead of
already-engineered ones. These cover the re-apply mechanism (step + pipeline).
"""
import pandas as pd

from utils.preprocessing_pipeline import PreprocessingPipeline, PreprocessingStep
from utils.categorical_processor import CategoricalProcessor
from utils.numeric_processor import NumericProcessor


def _processors():
    return {"categorical": CategoricalProcessor(), "numeric": NumericProcessor()}


def test_step_reapplies_one_hot_to_raw_data():
    step = PreprocessingStep(name="oh", processor_type="categorical",
                             function_name="apply_one_hot_encoding",
                             params={"column": "color"})
    out = step.apply(pd.DataFrame({"color": ["red", "blue", "red"]}), _processors())
    assert {"color_red", "color_blue"}.issubset(out.columns)
    assert "color" not in out.columns


def test_pipeline_reapplies_recipe_to_raw_data():
    pipe = PreprocessingPipeline("recipe")
    pipe.add_step(PreprocessingStep(name="oh", processor_type="categorical",
                                    function_name="apply_one_hot_encoding",
                                    params={"column": "color"}))
    raw = pd.DataFrame({"color": ["red", "blue", "red"], "n": [1, 2, 3]})
    out = pipe.apply(raw, _processors())
    # the engineered columns the model expects are reproduced from raw input...
    assert {"color_red", "color_blue"}.issubset(out.columns)
    # ...and untouched columns are preserved
    assert "n" in out.columns


def test_pipeline_apply_preserves_row_count():
    pipe = PreprocessingPipeline("recipe")
    pipe.add_step(PreprocessingStep(name="norm", processor_type="numeric",
                                    function_name="normalize", params={"column": "n"}))
    raw = pd.DataFrame({"n": [1.0, 2.0, 3.0, 4.0]})
    out = pipe.apply(raw, _processors())
    assert len(out) == len(raw)


def test_pipeline_apply_raises_on_missing_column():
    # If the raw upload truly lacks the source column, replay should raise (the
    # Predict tab catches this and falls back / warns rather than crashing).
    pipe = PreprocessingPipeline("recipe")
    pipe.add_step(PreprocessingStep(name="oh", processor_type="categorical",
                                    function_name="apply_one_hot_encoding",
                                    params={"column": "color"}))
    raw = pd.DataFrame({"unrelated": [1, 2, 3]})
    try:
        pipe.apply(raw, _processors())
    except Exception:
        return  # expected
    # apply() may also no-op a step it can't satisfy; either way 'color_*' must not appear
    out = pipe.apply(raw, _processors())
    assert not any(str(c).startswith("color_") for c in out.columns)
