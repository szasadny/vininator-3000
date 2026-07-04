"""Tests for eval/importance.py — SHAP ranking + figures for the rating model."""

from __future__ import annotations

import polars as pl

from vininator.config import get_settings
from vininator.eval.importance import run_shap_analysis
from vininator.models.rating import train_rating


def test_run_shap_analysis_smoke(processed_dataset, write_config, monkeypatch) -> None:
    """SHAP artifacts exist, rankings are finite, block totals reconcile."""
    # Figures must not land in the real repo's reports/ during tests.
    settings = get_settings()
    monkeypatch.setattr(
        type(settings), "figures_dir", property(lambda self: self.data_dir / "figures")
    )

    train_rating(write_config(), track=False)
    report = run_shap_analysis(split="test", sample_cells=200)

    assert report.n_cells > 0
    assert report.top_features, "expected a non-empty ranking"
    assert all(v >= 0.0 for _, v in report.top_features)

    importance = pl.read_parquet(report.importance_path)
    assert set(importance.columns) == {"feature", "block", "mean_abs_shap"}
    assert set(importance.get_column("block").unique().to_list()) <= {
        "terroir",
        "producer",
        "age",
        "wine",
    }
    # Block totals are exactly the per-feature sums regrouped.
    assert sum(report.block_totals.values()) == importance.get_column("mean_abs_shap").sum()

    # Summary figure + one dependence figure per configured feature present
    # in the model (all four exist in the synthetic table).
    assert len(report.figure_paths) == 5
    for fig in report.figure_paths:
        assert fig.exists() and fig.stat().st_size > 0
