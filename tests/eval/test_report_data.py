"""Tests for eval/report_data.py — the RESULTS.md data assembly."""

from __future__ import annotations

import polars as pl

from vininator.config import Settings
from vininator.eval.report_data import (
    age_well_display,
    baseline_grid,
    drink_now_display,
    grape_display,
    harmonize_eval,
    markdown_table,
    profile_eval,
    quantile_coverage,
    rating_band,
    readme_disclaimer,
    split_summary,
    value_display,
)


def test_markdown_table_formats_floats_none_and_lists() -> None:
    df = pl.DataFrame({"a": [1.23456, None], "b": [["x", "y"], []], "c": [True, False]})
    lines = markdown_table(df, floats="{:.2f}").splitlines()
    assert lines[0] == "| a | b | c |"
    assert lines[1] == "| --- | --- | --- |"
    assert lines[2] == "| 1.23 | x, y | true |"
    assert lines[3] == "|  |  | false |"


def _drink_now_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "winery_name": ["W1"],
            "wine_name": ["Wine 1"],
            "region_name": ["Bordeaux"],
            "vintage_year": [2018],
            "predicted_rating": [4.30],
            "predicted_rating_lo": [4.10],
            "predicted_rating_hi": [4.55],
            "predicted_body": ["Full-bodied"],
            "predicted_acidity": ["High"],
            "top_pairings": [["beef", "lamb"]],
        }
    )


def test_rating_band_formats_prediction_with_interval() -> None:
    assert rating_band(_drink_now_frame()) == ["4.30 (4.10-4.55)"]


def test_drink_now_display_headers_and_band() -> None:
    disp = drink_now_display(_drink_now_frame())
    assert disp.columns == [
        "Winery",
        "Wine",
        "Region",
        "Vintage",
        "Predicted (lo-hi)",
        "Body",
        "Acidity",
        "Pairings",
    ]
    assert disp.get_column("Predicted (lo-hi)").to_list() == ["4.30 (4.10-4.55)"]
    assert disp.get_column("Pairings").to_list() == ["beef, lamb"]


def test_age_well_display_headers() -> None:
    frame = pl.DataFrame(
        {
            "winery_name": ["W1"],
            "wine_name": ["Wine 1"],
            "region_name": ["Rioja"],
            "vintage_year": [2019],
            "predicted_peak_year": [2030],
            "predicted_peak_rating": [4.4],
            "slope_to_peak": [0.12],
            "trajectory": ["rising"],
            "age_clipped_any": [False],
        }
    )
    disp = age_well_display(frame)
    assert disp.columns == [
        "Winery",
        "Wine",
        "Region",
        "Vintage",
        "Peak yr",
        "Peak",
        "Slope/yr",
        "Trajectory",
        "Clipped",
    ]
    assert disp.get_column("Slope/yr").to_list() == ["+0.120"]


def test_value_display_headers_and_price_formatting() -> None:
    frame = _drink_now_frame().with_columns(
        pl.Series("price_eur", [25.0]),
        pl.Series("price_band", ["mid"]),
        pl.Series("match_confidence", ["exact"]),
    )
    disp = value_display(frame)
    assert disp.columns == [
        "Winery",
        "Wine",
        "Region",
        "Vintage",
        "Predicted (lo-hi)",
        "Price (EUR)",
        "Band",
        "Rating/€10",
        "Match",
    ]
    assert disp.get_column("Price (EUR)").to_list() == ["€25"]
    # 4.30 / 25 * 10 = 1.72
    assert disp.get_column("Rating/€10").to_list() == ["1.72"]


def test_grape_display_titlecases_slug() -> None:
    assert grape_display("cabernet-sauvignon") == "Cabernet Sauvignon"


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
