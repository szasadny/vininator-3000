"""Tests for recommend/drink_now.py — candidate table, filters, and scoring."""

from __future__ import annotations

import polars as pl

from vininator.config import Settings
from vininator.recommend.drink_now import (
    DRINK_NOW_OUT_COLS,
    RecommendFilters,
    apply_filters,
    build_candidates,
    load_recommend_bundles,
    recommend_drink_now,
    train_age_bounds,
)


def test_build_candidates_unique_and_named(processed_dataset: Settings) -> None:
    candidates = build_candidates()

    # One row per (wine, vintage); names + monogrape flag joined from raw wines.
    assert candidates.select("wine_id", "vintage_year").is_duplicated().sum() == 0
    for col in ("wine_name", "winery_name", "is_monogrape", "grape_majority", "split"):
        assert col in candidates.columns
    assert candidates.get_column("wine_name").null_count() == 0
    assert candidates.get_column("is_monogrape").null_count() == 0
    # The observed cell mean is carried as metadata, not the model target name.
    assert "observed_mean_rating" in candidates.columns
    assert "rating" not in candidates.columns


def test_apply_filters_grape_and_monogrape(processed_dataset: Settings) -> None:
    candidates = build_candidates()
    # All synthetic wines are single-varietal, so monogrape keeps everything.
    assert (
        apply_filters(candidates, RecommendFilters(monogrape=True), 2026).height
        == candidates.filter(pl.col("vintage_year") <= 2026).height
    )

    pinot = apply_filters(candidates, RecommendFilters(grape="pinot-noir"), 2026)
    assert pinot.height > 0
    assert (pinot.get_column("grape_majority").str.to_lowercase() == "pinot noir").all()


def test_max_vintage_age_boundary(processed_dataset: Settings) -> None:
    candidates = build_candidates()
    # At opening 2026 the only vintage within 6 years is 2020 (age 6); a cap of
    # 5 excludes even that, so nothing survives.
    within_6 = apply_filters(candidates, RecommendFilters(max_vintage_age=6), 2026)
    assert within_6.height > 0
    assert (within_6.get_column("vintage_year") == 2020).all()
    assert apply_filters(candidates, RecommendFilters(max_vintage_age=5), 2026).is_empty()


def test_recommend_drink_now_scores_and_writes(trained_bundles: Settings) -> None:
    report = recommend_drink_now(opening_year=2026, top=5)

    assert report.path.exists()
    assert report.table.columns == list(DRINK_NOW_OUT_COLS)
    assert 0 < report.table.height <= 5
    # Sorted by predicted rating descending.
    ratings = report.table.get_column("predicted_rating").to_list()
    assert ratings == sorted(ratings, reverse=True)
    assert report.table.get_column("predicted_rating").is_between(1.0, 5.5).all()

    # Every synthetic vintage (<= 2020) is far past the tight train age range,
    # so scoring at 2026 clips the age on every row.
    lo, hi = train_age_bounds()
    assert (2026 - report.table.get_column("vintage_year")).min() > hi
    assert report.table.get_column("age_clipped").all()
    assert (report.table.get_column("age_at_review") == hi).all()

    written = pl.read_parquet(report.path)
    assert written.height == report.table.height


def test_shared_candidates_and_bundles_match_fresh_run(trained_bundles: Settings) -> None:
    # Passing a prebuilt candidate table + bundles must give the identical ranking
    # to letting the function build them itself (the generator reuses them).
    cands = build_candidates()
    bundles = load_recommend_bundles()
    shared = recommend_drink_now(opening_year=2026, top=5, candidates=cands, bundles=bundles)
    fresh = recommend_drink_now(opening_year=2026, top=5)

    assert (
        shared.table.get_column("wine_id").to_list() == fresh.table.get_column("wine_id").to_list()
    )
    assert (
        shared.table.get_column("predicted_rating").to_list()
        == fresh.table.get_column("predicted_rating").to_list()
    )
