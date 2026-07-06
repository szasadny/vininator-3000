"""Tests for recommend/standout_years.py — per-year shortlists reuse drink-now."""

from __future__ import annotations

from vininator.config import Settings
from vininator.recommend.drink_now import RecommendFilters, recommend_drink_now
from vininator.recommend.standout_years import recommend_standout_years


def test_standout_year_matches_standalone_drink_now(trained_bundles: Settings) -> None:
    """A year's shortlist must equal the standalone drink-now ranking for that year."""
    filters = RecommendFilters()
    standout = recommend_standout_years(from_year=2026, to_year=2026, filters=filters, top=5)
    drink_now = recommend_drink_now(opening_year=2026, filters=filters, top=5)

    year_block = standout.table.filter(standout.table["opening_year"] == 2026).sort("rank")
    assert (
        year_block.get_column("wine_id").to_list()
        == drink_now.table.get_column("wine_id").to_list()
    )


def test_standout_rank_is_dense_per_year(trained_bundles: Settings) -> None:
    report = recommend_standout_years(from_year=2026, to_year=2028, top=4)

    assert report.path.exists()
    assert set(report.years_covered) == {2026, 2027, 2028}
    for year in report.years_covered:
        block = report.table.filter(report.table["opening_year"] == year).sort("rank")
        ranks = block.get_column("rank").to_list()
        assert ranks == list(range(1, len(ranks) + 1))
        assert len(ranks) <= 4
