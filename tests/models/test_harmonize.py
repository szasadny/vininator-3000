"""Tests for models/harmonize.py — multilabel model + tuned thresholds."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from vininator.config import Settings
from vininator.models.artifacts import load_bundle
from vininator.models.harmonize import train_harmonize


def test_train_harmonize_tunes_thresholds_and_reports(
    processed_dataset: Settings, write_config: Callable[..., Path]
) -> None:
    report = train_harmonize(write_config("harmonize"), track=False)

    assert report.n_labels > 0
    assert report.bundle_names == ["harmonize"]

    # Both eval splits must be present in the report.
    split_names = {sm.split for sm in report.metrics}
    assert split_names == {"test", "future_vintage_test"}

    for sm in report.metrics:
        assert len(sm.per_label_f1) == report.n_labels
        assert 0.0 <= sm.hamming <= 1.0

    bundle = load_bundle("harmonize")
    assert len(bundle.meta["thresholds"]) == report.n_labels
    assert len(bundle.meta["targets"]) == report.n_labels
    assert all(0.0 <= t <= 1.0 for t in bundle.meta["thresholds"].values())
    # Saved meta must carry both splits.
    assert "test" in bundle.meta["metrics"]
    assert "future_vintage_test" in bundle.meta["metrics"]
