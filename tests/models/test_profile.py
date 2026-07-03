"""Tests for models/profile.py — Body + Acidity classifiers."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from vininator.config import Settings
from vininator.models.artifacts import load_bundle
from vininator.models.profile import train_profile


def test_train_profile_reports_macro_f1_and_saves_both_bundles(
    processed_dataset: Settings, write_config: Callable[..., Path]
) -> None:
    report = train_profile(write_config("profile"), track=False)

    # Both targets must appear on both eval splits.
    split_target_pairs = {(m.split, m.target) for m in report.metrics}
    for split in ("test", "future_vintage_test"):
        for target in ("body_label", "acidity_label"):
            assert (split, target) in split_target_pairs, (
                f"missing ({split!r}, {target!r}) in profile metrics"
            )

    for m in report.metrics:
        assert 0.0 <= m.macro_f1 <= 1.0
        assert 0.0 <= m.accuracy <= 1.0

    for name in ("body", "acidity"):
        assert name in report.bundle_names
        bundle = load_bundle(name)
        assert bundle.meta["model_class"] == "CatBoostClassifier"
        assert len(bundle.meta["classes"]) >= 2
        # Saved meta must carry both splits so the report is reproducible from disk.
        assert "test" in bundle.meta["metrics"]
        assert "future_vintage_test" in bundle.meta["metrics"]
