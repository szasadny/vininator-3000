"""Atomic save/load for trained model bundles.

A *bundle* is a CatBoost model file (`<name>.cbm`) plus a sidecar
(`<name>.meta.json`) carrying everything the recommender (Phase 6) needs to
score a wine without re-deriving it: the feature list, the categorical subset,
the target(s), per-label thresholds, the training config, and provenance
(dataset hash, git SHA, metrics). Both files are written tmp→rename so a crash
mid-save never leaves a half-written model behind — the same idiom used by
features/build.py.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

from catboost import CatBoost, CatBoostClassifier, CatBoostRegressor

from vininator.config import get_settings
from vininator.models.dataset import FeatureSpec

_MODEL_CLASSES: dict[str, type[CatBoost]] = {
    "CatBoostRegressor": CatBoostRegressor,
    "CatBoostClassifier": CatBoostClassifier,
}


@dataclasses.dataclass(frozen=True)
class ModelBundle:
    """A loaded model and its metadata sidecar."""

    model: CatBoost
    meta: dict[str, Any]

    def feature_spec(self) -> FeatureSpec:
        """Rebuild the `FeatureSpec` the model was trained with."""
        return FeatureSpec(
            feature_cols=list(self.meta["feature_cols"]),
            cat_cols=list(self.meta["cat_cols"]),
        )


def _atomic_write(target: Path, write: Any) -> None:
    """Write via a `.tmp` sibling, then atomically rename onto `target`."""
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    write(tmp)
    tmp.replace(target)


def bundle_paths(name: str, *, models_dir: Path | None = None) -> tuple[Path, Path]:
    """Return `(cbm_path, meta_path)` for a bundle stem."""
    root = models_dir if models_dir is not None else get_settings().models_dir
    return root / f"{name}.cbm", root / f"{name}.meta.json"


def save_bundle(
    model: CatBoost,
    meta: dict[str, Any],
    *,
    name: str,
    models_dir: Path | None = None,
) -> tuple[Path, Path]:
    """Persist a model + its metadata sidecar atomically.

    `meta` must include `model_class` (one of the keys in `_MODEL_CLASSES`) so
    `load_bundle` can reconstruct the right CatBoost subclass.
    """
    cbm_path, meta_path = bundle_paths(name, models_dir=models_dir)
    _atomic_write(cbm_path, lambda p: model.save_model(str(p), format="cbm"))
    _atomic_write(
        meta_path,
        lambda p: p.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8"),
    )
    return cbm_path, meta_path


def load_bundle(name: str, *, models_dir: Path | None = None) -> ModelBundle:
    """Load a bundle saved by `save_bundle`.

    The CatBoost subclass is taken from `meta["model_class"]`; an unknown or
    missing value falls back to the base `CatBoost`, which can still predict.
    """
    cbm_path, meta_path = bundle_paths(name, models_dir=models_dir)
    if not cbm_path.exists():
        raise FileNotFoundError(f"No model bundle named {name!r} at {cbm_path}.")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    model_cls = _MODEL_CLASSES.get(meta.get("model_class", ""), CatBoost)
    model = model_cls()
    model.load_model(str(cbm_path))
    return ModelBundle(model=model, meta=meta)
