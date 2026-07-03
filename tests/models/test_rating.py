"""Tests for models/rating.py — trains a tiny regressor on synthetic data."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from vininator.config import Settings
from vininator.models.artifacts import load_bundle
from vininator.models.dataset import build_pool, load_split
from vininator.models.rating import train_rating


def test_train_rating_produces_metrics_baselines_and_bundles(
    processed_dataset: Settings, write_config: Callable[..., Path]
) -> None:
    report = train_rating(write_config("rating"), track=False)

    assert report.n_train > 0
    assert report.n_features > 0
    splits = {m.split for m in report.eval_metrics}
    assert "test" in splits
    # Both evaluation levels are reported, plus the per-rating noise floor.
    for m in report.eval_metrics:
        assert m.rmse >= 0.0
        assert m.cell_rmse >= 0.0
        assert m.cell_mae >= 0.0
        assert m.noise_floor >= 0.0
    # Every evaluated split carries the full four-baseline grid at both levels.
    for split in splits:
        assert len(report.baselines[split]) == 4
        assert len(report.cell_baselines[split]) == 4

    for name in ("rating", "rating_q_lo", "rating_q_hi"):
        assert name in report.bundle_names
        assert (processed_dataset.models_dir / f"{name}.cbm").exists()
        assert (processed_dataset.models_dir / f"{name}.meta.json").exists()


def test_rating_meta_carries_both_metric_levels(
    processed_dataset: Settings, write_config: Callable[..., Path]
) -> None:
    """The saved bundle must be self-describing about how it was measured."""
    train_rating(write_config("rating"), track=False)
    meta = load_bundle("rating").meta
    for split_metrics in meta["metrics"].values():
        assert {"rmse", "mae", "cell_rmse", "cell_mae", "noise_floor"} <= set(split_metrics)


def test_quantile_band_is_ordered_on_average(
    processed_dataset: Settings, write_config: Callable[..., Path]
) -> None:
    train_rating(write_config("rating"), track=False)
    lo = load_bundle("rating_q_lo")
    hi = load_bundle("rating_q_hi")

    test_df = load_split("test")
    pool = build_pool(test_df, lo.feature_spec(), label=None)
    pred_lo = lo.model.predict(pool)
    pred_hi = hi.model.predict(pool)
    assert pred_lo.mean() <= pred_hi.mean() + 1e-6
