"""Tests for eval/ablations.py — the Phase 5 feature-block ablation runner."""

from __future__ import annotations

import pytest

from vininator.config import get_settings
from vininator.eval.ablations import ablation_arms, run_ablations
from vininator.features.terroir import terroir_feature_cols


def test_arms_cover_the_projectmd_blocks() -> None:
    """The arm grid is exactly full + the three PROJECT.md ablation blocks."""
    arms = ablation_arms()
    assert list(arms) == ["full", "no_terroir", "no_producer", "no_age"]
    assert arms["full"] == ()
    assert set(arms["no_terroir"]) == set(terroir_feature_cols())
    assert arms["no_age"] == ("age_at_review",)


def test_unknown_arm_raises(processed_dataset, write_config) -> None:
    with pytest.raises(ValueError, match="Unknown ablation arm"):
        run_ablations(write_config(), arms=["no_such_block"], track=False)


def test_run_ablations_smoke(processed_dataset, write_config) -> None:
    """Two arms train, evaluate both splits, and leave no scratch behind."""
    settings = get_settings()
    report = run_ablations(write_config(), arms=["full", "no_terroir"], track=False)

    by_arm = {r.arm: r for r in report.results}
    assert set(by_arm) == {"full", "no_terroir"}
    # Dropping the terroir block must shrink the feature set by exactly the
    # block size (every terroir column is present in the synthetic table).
    assert by_arm["full"].n_features - by_arm["no_terroir"].n_features == len(
        terroir_feature_cols()
    )
    for result in report.results:
        splits = {m.split for m in result.metrics}
        assert splits == {"test", "future_vintage_test"}
        for m in result.metrics:
            assert m.cell_rmse > 0.0

    # The grid parquet exists and has one row per (arm, split).
    assert report.parquet_path.exists()
    import polars as pl

    grid = pl.read_parquet(report.parquet_path)
    assert grid.height == 4
    assert set(grid.get_column("arm").to_list()) == {"full", "no_terroir"}

    # The shared cells scratch parquet is cleaned up.
    leftovers = list(settings.snapshots_dir.glob("ablation_cells_*.parquet"))
    assert leftovers == []
