"""Tests for recommend/age_well.py — trajectory summary + sweep shape."""

from __future__ import annotations

import polars as pl

from vininator.config import Settings
from vininator.recommend.age_well import _summarize, recommend_age_well


def _long_row(wine_id: int, year: int, rating: float) -> dict:
    return {
        "wine_id": wine_id,
        "vintage_year": 2020,
        "winery_name": f"Winery {wine_id}",
        "wine_name": f"Wine {wine_id}",
        "region_name": "Bordeaux",
        "opening_year": year,
        "predicted_rating": rating,
        "age_clipped": False,
    }


def test_summarize_classifies_trajectories() -> None:
    years = [2026, 2027, 2028, 2029]
    trajectories = {
        1: [3.0, 3.5, 4.0, 3.8],  # peak in the interior -> peaks_late
        2: [3.0, 3.2, 3.4, 3.6],  # peak at the end -> rising
        3: [3.0, 3.0, 3.01, 3.0],  # flat within eps -> plateau
        4: [3.6, 3.4, 3.2, 3.0],  # peak at the start -> declining
    }
    rows = [
        _long_row(wid, year, rating)
        for wid, ratings in trajectories.items()
        for year, rating in zip(years, ratings, strict=True)
    ]
    summary = _summarize(pl.DataFrame(rows), opening_year=2026).sort("wine_id")

    by_wine = {r["wine_id"]: r for r in summary.iter_rows(named=True)}
    assert by_wine[1]["trajectory"] == "peaks_late"
    assert by_wine[1]["predicted_peak_year"] == 2028
    assert by_wine[1]["predicted_peak_rating"] == 4.0
    assert by_wine[1]["slope_to_peak"] == (4.0 - 3.0) / (2028 - 2026)

    assert by_wine[2]["trajectory"] == "rising"
    assert by_wine[2]["predicted_peak_year"] == 2029

    assert by_wine[3]["trajectory"] == "plateau"

    assert by_wine[4]["trajectory"] == "declining"
    assert by_wine[4]["predicted_peak_year"] == 2026


def test_recommend_age_well_sweep_shape(trained_bundles: Settings) -> None:
    horizon = 3
    report = recommend_age_well(opening_year=2026, horizon=horizon, top=10)

    assert report.long_path.exists()
    assert report.summary_path.exists()

    # Every bottle is scored at every year in the horizon window.
    per_bottle = report.long.group_by("wine_id", "vintage_year").len()
    assert per_bottle.get_column("len").unique().to_list() == [horizon + 1]
    n_bottles = report.long.select("wine_id", "vintage_year").n_unique()
    assert report.summary.height == n_bottles

    # The ranked table excludes declining bottles and is a subset of the summary.
    assert (report.table.get_column("trajectory") != "declining").all()
    assert report.table.height <= report.summary.height
