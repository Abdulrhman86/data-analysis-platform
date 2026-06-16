"""Phase 6 — model export / reload / predict round-trip.

The headline "train a model, export it, reload it elsewhere, predict on raw data"
story was previously untested. These exercise it end to end.
"""
import base64
import io
import json
import os
import zipfile

import numpy as np
import pandas as pd

from utils.model_export import (
    save_model, load_model, create_download_link, export_full_pipeline, _json_default,
)
from utils.classification_models import ClassificationProcessor


def _trained():
    rng = np.random.RandomState(0)
    df = pd.DataFrame({
        "num": rng.rand(40),
        "cat": rng.choice(["a", "b"], 40),   # categorical -> exercises the encoder
        "target": rng.choice([0, 1], 40),
    })
    proc = ClassificationProcessor()
    split = proc.prepare_data(df, "target", test_size=0.3)
    name = list(proc.models.keys())[0]
    model = proc.train_model(name, split["X_train"], split["y_train"])
    return model, split


def test_save_load_predict_round_trip(tmp_path):
    model, split = _trained()
    before = model.predict(split["X_test"])
    path = save_model(model, "My Model", directory=str(tmp_path))
    assert os.path.exists(path)
    reloaded = load_model(path)
    after = reloaded.predict(split["X_test"])
    # reloaded model predicts identically on RAW (un-pre-scaled) input
    assert list(before) == list(after)


def test_create_download_link_bundles_model_and_metadata():
    model, _ = _trained()
    href = create_download_link(model, "My Model", metadata={"acc": 0.9})
    b64 = href.split("base64,")[1].split('"')[0]
    zf = zipfile.ZipFile(io.BytesIO(base64.b64decode(b64)))
    assert "model.pkl" in zf.namelist()
    assert "metadata.json" in zf.namelist()


def test_export_full_pipeline_bundles_everything():
    model, _ = _trained()
    steps = [{"name": "step1", "function_name": "normalize"}]
    buf = export_full_pipeline({"My Model": model}, {"task": "classification"}, steps)
    names = zipfile.ZipFile(buf).namelist()
    assert any(n.startswith("models/") and n.endswith(".pkl") for n in names)
    assert "data_info.json" in names
    assert "preprocessing_steps.json" in names


def test_numpy_metadata_is_json_safe():
    meta = {"cm": np.array([[1, 2], [3, 4]]), "n": np.int64(5), "f": np.float64(0.5)}
    json.dumps(meta, default=_json_default)   # must not raise
