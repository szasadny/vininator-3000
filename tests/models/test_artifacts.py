"""Tests for models/artifacts.py — bundle round-trip and atomic write."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from vininator.models.artifacts import load_bundle, save_bundle


def _tiny_model() -> tuple[CatBoostRegressor, pd.DataFrame]:
    x = pd.DataFrame({"a": [0, 1, 2, 3, 4, 5], "c": ["x", "y", "x", "y", "x", "y"]})
    y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    model = CatBoostRegressor(iterations=5, depth=2, verbose=False, allow_writing_files=False)
    model.fit(x, y, cat_features=["c"])
    return model, x


def test_bundle_roundtrip_predicts_identically(tmp_path: Path) -> None:
    model, x = _tiny_model()
    meta = {
        "model_class": "CatBoostRegressor",
        "feature_cols": ["a", "c"],
        "cat_cols": ["c"],
    }
    save_bundle(model, meta, name="t", models_dir=tmp_path)
    loaded = load_bundle("t", models_dir=tmp_path)

    assert np.allclose(loaded.model.predict(x), model.predict(x))
    assert loaded.meta["model_class"] == "CatBoostRegressor"
    assert loaded.feature_spec().cat_cols == ["c"]


def test_save_leaves_no_tmp_files(tmp_path: Path) -> None:
    model, _ = _tiny_model()
    save_bundle(model, {"model_class": "CatBoostRegressor"}, name="t", models_dir=tmp_path)
    assert list(tmp_path.glob("*.tmp")) == []
    assert (tmp_path / "t.cbm").exists()
    assert (tmp_path / "t.meta.json").exists()
