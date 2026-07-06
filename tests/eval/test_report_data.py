"""Tests for eval/report_data.py — the RESULTS.md data assembly."""

from __future__ import annotations

import polars as pl

from vininator.config import Settings
from vininator.eval.report_data import (
    baseline_grid,
    harmonize_eval,
    markdown_table,
    profile_eval,
    quantile_coverage,
    readme_disclaimer,
    split_summary,
)


def test_markdown_table_formats_floats_none_and_lists() -> None:
    df = pl.DataFrame({"a": [1.23456, None], "b": [["x", "y"], []], "c": [True, False]})
    lines = markdown_table(df, floats="{:.2f}").splitlines()
    assert lines[0] == "| a | b | c |"
    assert lines[1] == "| --- | --- | --- |"
    assert lines[2] == "| 1.23 | x, y | true |"
    assert lines[3] == "|  |  | false |"


def test_split_summary_covers_all_three_splits(processed_dataset: Settings) -> None:
    summary = split_summary()
    assert summary.get_column("split").to_list() == ["train", "test", "future_vintage_test"]
    assert (summary.get_column("ratings") > 0).all()
    assert (summary.get_column("wine_vintages") <= summary.get_column("ratings")).all()


def test_baseline_grid_has_model_and_beats_global(trained_bundles: Settings) -> None:
    grids = baseline_grid()
    assert set(grids) == {"test", "future_vintage_test"}
    test = grids["test"]
    assert test.get_column("predictor").to_list() == [
        "model (CatBoost)",
        "global_mean",
        "winery_mean",
        "region_vintage_mean",
        "grape_region_mean",
        "noise_floor",
    ]
    # The synthetic data's rating is a function of winery, so the per-winery
    # baseline must not be worse than predicting the global mean.
    by_name = {r["predictor"]: r for r in test.iter_rows(named=True)}
    assert by_name["winery_mean"]["cell_rmse"] <= by_name["global_mean"]["cell_rmse"]
    # The noise-floor row is per-rating only; its cell columns stay blank.
    assert by_name["noise_floor"]["cell_rmse"] is None


def test_quantile_coverage_in_unit_interval(trained_bundles: Settings) -> None:
    coverage = quantile_coverage()
    assert coverage.get_column("split").to_list() == ["test", "future_vintage_test"]
    assert coverage.get_column("coverage").is_between(0.0, 1.0).all()
    assert (coverage.get_column("n_cells") > 0).all()


def test_profile_eval_confusion_is_square_over_classes(trained_bundles: Settings) -> None:
    report = profile_eval("body_label")
    n = len(report.classes)
    # Confusion has a label column plus one column per class, and one row per class.
    assert report.confusion.height == n
    assert report.confusion.width == n + 1
    assert report.per_class_f1.get_column("class").to_list() == report.classes
    assert report.per_class_f1.get_column("f1").is_between(0.0, 1.0).all()


def test_harmonize_eval_labels_and_examples(trained_bundles: Settings) -> None:
    report = harmonize_eval()
    assert report.per_label_f1.get_column("f1").is_between(0.0, 1.0).all()
    assert report.headline.get_column("hamming").is_between(0.0, 1.0).all()
    assert 0 < report.examples.height <= 3
    assert {"wine", "region", "vintage", "predicted", "actual"} <= set(report.examples.columns)


def test_readme_disclaimer_extracts_the_section() -> None:
    text = readme_disclaimer()
    assert "hobby / learning" in text
    # The extractor must stop before the next top-level heading.
    assert "\n## " not in text
