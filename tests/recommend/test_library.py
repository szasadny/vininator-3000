"""Tests for recommend/library.py — catalog filtering, price join, value views.

The pure functions are tested on hand-built frames; `build_catalog_section` and
`vintage_quality` run against the synthetic `trained_bundles` fixture (whose
vintages include 2020, so the library window `vintage_year > 2018` is non-empty).
"""

from __future__ import annotations

import polars as pl
import pytest

from vininator.config import (
    DEFAULT_OPENING_YEAR,
    PRICE_APPRECIATION_ANNUAL,
    PRICE_INFLATION_ANNUAL,
    PRICE_SNAPSHOT_YEAR,
    PRICE_SOURCE,
    PRICE_SOURCE_CURRENCY,
    PRICE_USD_PER_EUR,
    VALUE_PRICE_CAP_EUR,
    Settings,
)
from vininator.recommend.drink_now import build_candidates, load_recommend_bundles
from vininator.recommend.library import (
    LIBRARY_DRINK_NOW_COLS,
    _current_price_eur_expr,
    _price_band_expr,
    build_catalog_section,
    favorite_slices,
    filter_library_candidates,
    filter_vintage_window,
    future_greats,
    join_price,
    list_grape_groups,
    value_views,
    vintage_quality,
)


def _candidates() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "wine_id": [1, 2, 3, 4, 5, 6],
            "vintage_year": [2015, 2018, 2019, 2020, 2020, 2019],
            "is_monogrape": [True, True, True, False, True, True],
            "wine_type": ["Red", "White", "Red", "Red", "Sparkling", "Red"],
            "grape_majority": ["Merlot", "Riesling", "Merlot", None, "Chardonnay", "Merlot"],
        }
    )


def test_filter_vintage_window_excludes_old() -> None:
    window = filter_vintage_window(_candidates())
    # Window is vintage_year > 2018, so 2015 and 2018 both drop; 2019/2020 stay.
    assert sorted(window.get_column("wine_id").to_list()) == [3, 4, 5, 6]


def test_filter_library_candidates_drops_blends_types_and_null_grape() -> None:
    general = filter_library_candidates(filter_vintage_window(_candidates()))
    # Keeps monogrape Red/White with a grape: wines 3, 6. Drops the blend (4), the
    # Sparkling (5), and the pre-window wines (1 @2015, 2 @2018).
    assert sorted(general.get_column("wine_id").to_list()) == [3, 6]


def test_list_grape_groups_threshold_and_ordering() -> None:
    general = filter_library_candidates(filter_vintage_window(_candidates()))
    groups = list_grape_groups(general, min_bottles=2)
    # Only (Red, Merlot) clears the threshold (wines 3, 6); the White Riesling
    # (wine 2 @2018) is out of the vintage window entirely.
    assert groups.get_column("grape_majority").to_list() == ["Merlot"]
    assert groups.get_column("n_bottles").to_list() == [2]

    # Type ordering Red -> White -> Rosé, then count desc within a type.
    wide = pl.DataFrame(
        {
            "wine_id": range(1, 8),
            "vintage_year": [2020] * 7,
            "is_monogrape": [True] * 7,
            "wine_type": ["White", "White", "Red", "Red", "Red", "Rosé", "Rosé"],
            "grape_majority": [
                "Riesling",
                "Riesling",
                "Merlot",
                "Merlot",
                "Syrah",
                "Grenache",
                "Grenache",
            ],
        }
    )
    ordered = list_grape_groups(wide, min_bottles=1)
    assert ordered.get_column("wine_type").to_list()[:1] == ["Red"]
    assert ordered.get_column("wine_type").to_list()[-1] == "Rosé"
    # Within Red, Merlot (2) sorts before Syrah (1).
    red = ordered.filter(pl.col("wine_type") == "Red")
    assert red.get_column("grape_majority").to_list() == ["Merlot", "Syrah"]


def test_join_price_none_path_all_null() -> None:
    joined = join_price(_candidates(), None)
    for col in ("price_estimate", "price_source", "price_currency", "price_eur", "price_band"):
        assert col in joined.columns
    assert (joined.get_column("match_confidence") == "none").all()
    assert joined.get_column("price_eur").null_count() == joined.height
    assert joined.get_column("price_band").null_count() == joined.height


def test_join_price_adjusts_and_bands() -> None:
    price = pl.DataFrame(
        {
            "wine_id": [2, 3, 6],  # vintages 2018, 2019, 2019
            "price_estimate": [100.0, 100.0, 8.0],
            "price_source": [PRICE_SOURCE] * 3,
            "price_currency": [PRICE_SOURCE_CURRENCY] * 3,
            "match_confidence": ["exact", "winery-median", "exact"],
        }
    )
    joined = join_price(_candidates(), price).sort("wine_id")
    by_id = {r["wine_id"]: r for r in joined.iter_rows(named=True)}
    # Same 100 USD: wine 2 (vintage 2018) is a year older than wine 3 (2019), so
    # its aging premium is larger -> higher adjusted EUR price.
    assert by_id[2]["price_eur"] > by_id[3]["price_eur"]
    # Adjustment lifts price above the naive USD/FX floor (inflation + aging > 1).
    assert by_id[3]["price_eur"] > 100.0 / PRICE_USD_PER_EUR
    # Cheap wine lands low, expensive wine high, after banding on the adjusted EUR.
    assert by_id[6]["price_band"] in {"budget", "mid"}
    assert by_id[2]["price_band"] in {"premium", "cult"}
    # Unmatched wines keep null price + "none".
    assert by_id[1]["match_confidence"] == "none"
    assert by_id[1]["price_eur"] is None


def test_price_band_expr_boundaries() -> None:
    df = pl.DataFrame({"price_eur": [14.99, 15.0, 39.99, 40.0, 149.0, 150.0, None]})
    banded = df.with_columns(_price_band_expr(pl.col("price_eur")).alias("band"))
    assert banded.get_column("band").to_list() == [
        "budget",
        "mid",
        "mid",
        "premium",
        "premium",
        "cult",
        None,
    ]


def test_current_price_eur_adjustment_isolates_factors() -> None:
    # Row A: vintage == opening year -> age 0 -> only inflation + FX, no aging.
    # Row B: a 2021 bottle (age 5 in 2026) -> inflation + 5 years of aging + FX.
    df = pl.DataFrame(
        {"price_estimate": [100.0, 100.0], "vintage_year": [DEFAULT_OPENING_YEAR, 2021]}
    )
    out = df.with_columns(
        _current_price_eur_expr(pl.col("price_estimate"), pl.col("vintage_year")).alias("eur")
    )
    inflation = (1.0 + PRICE_INFLATION_ANNUAL) ** (DEFAULT_OPENING_YEAR - PRICE_SNAPSHOT_YEAR)
    eur = out.get_column("eur").to_list()
    assert eur[0] == pytest.approx(100.0 * inflation / PRICE_USD_PER_EUR, rel=1e-9)
    aging = (1.0 + PRICE_APPRECIATION_ANNUAL) ** 5
    assert eur[1] == pytest.approx(100.0 * inflation * aging / PRICE_USD_PER_EUR, rel=1e-9)


def _scored(prices: list[float | None]) -> pl.DataFrame:
    n = len(prices)
    return pl.DataFrame(
        {
            "wine_id": list(range(1, n + 1)),
            "vintage_year": [2020] * n,
            "predicted_rating": [4.5, 4.0, 3.0][:n] + [3.5] * max(0, n - 3),
            "price_eur": prices,
        }
    )


def test_value_views_excludes_nulls_and_respects_cap() -> None:
    scored = _scored([10.0, 100.0, None, 20.0])  # ratings 4.5, 4.0, 3.0, 3.5
    under_cap, by_value = value_views(scored, top=10, cap_eur=30.0)
    # Under cap: only the <=30 EUR priced wines (1 @10, 4 @20), sorted by rating.
    assert under_cap.get_column("wine_id").to_list() == [1, 4]
    # By value: rating/price -> wine1 0.45, wine4 0.175, wine2 0.04; null dropped.
    assert by_value.get_column("wine_id").to_list() == [1, 4, 2]
    assert "value_score" in by_value.columns


def test_future_greats_filters_and_sorts() -> None:
    summary = pl.DataFrame(
        {
            "wine_id": [1, 2, 3, 4],
            "vintage_year": [2020, 2020, 2020, 2020],
            "predicted_peak_year": [2031, 2028, 2033, 2030],
            "predicted_peak_rating": [4.2, 4.9, 4.4, 4.6],
            "trajectory": ["rising", "rising", "declining", "peaks_late"],
        }
    )
    fg = future_greats(summary, top=10, min_peak_year=2030)
    # Wine 2 excluded (peaks 2028 < 2030); wine 3 excluded (declining, though 2033).
    assert fg.get_column("wine_id").to_list() == [4, 1]  # by peak rating desc: 4.6, 4.2


def test_favorite_slices_grapes_and_region_styles() -> None:
    window = pl.DataFrame(
        {
            "wine_id": [1, 2, 3, 4, 5],
            "vintage_year": [2020] * 5,
            "is_monogrape": [True, True, False, True, True],
            "wine_type": ["Red", "Red", "Red", "White", "Red"],
            "grape_majority": ["Sangiovese", "Corvina", "Corvina", "Sangiovese", "Primitivo"],
            "region_name": [
                "Toscana",
                "Amarone della Valpolicella",
                "Amarone della Valpolicella",
                "Toscana",
                "Primitivo di Manduria",
            ],
        }
    )
    slices = dict(favorite_slices(window))
    # Sangiovese grape slice: monogrape only -> wines 1 and 4 (white counts too).
    assert sorted(slices["Sangiovese"].get_column("wine_id").to_list()) == [1, 4]
    # Amarone style: Red region match, blends included -> wines 2 and 3.
    assert sorted(slices["Amarone della Valpolicella"].get_column("wine_id").to_list()) == [2, 3]


# ---------------------------------------------------------------------------
# Integration against the synthetic trained bundles
# ---------------------------------------------------------------------------


def test_build_catalog_section_unpriced(trained_bundles: Settings) -> None:
    settings = trained_bundles
    window = join_price(filter_vintage_window(build_candidates()), None)
    general = filter_library_candidates(window)
    groups = list_grape_groups(general, min_bottles=1)
    assert groups.height > 0

    row = groups.row(0, named=True)
    slice_ = general.filter(
        (pl.col("wine_type") == row["wine_type"])
        & (pl.col("grape_majority") == row["grape_majority"])
    )
    section = build_catalog_section(
        row["wine_type"],
        row["grape_majority"],
        slice_,
        load_recommend_bundles(),
        settings.library_tables_dir,
        top=5,
    )
    assert section.drink_now.columns == list(LIBRARY_DRINK_NOW_COLS)
    assert 0 < section.drink_now.height <= 5
    assert section.n_bottles > 0
    # No price snapshot -> value views empty, and drink-now prices are null.
    assert section.best_value.is_empty()
    assert section.drink_now.get_column("price_eur").null_count() == section.drink_now.height
    # Age-well parquet was written under the tables dir.
    assert list(settings.library_tables_dir.glob("*_age_well.parquet"))


def test_build_catalog_section_priced(trained_bundles: Settings) -> None:
    settings = trained_bundles
    window = filter_vintage_window(build_candidates())
    ids = window.get_column("wine_id").unique().to_list()
    price = pl.DataFrame(
        {
            "wine_id": ids,
            "price_estimate": [25.0] * len(ids),
            "price_source": [PRICE_SOURCE] * len(ids),
            "price_currency": [PRICE_SOURCE_CURRENCY] * len(ids),
            "match_confidence": ["exact"] * len(ids),
        }
    )
    general = filter_library_candidates(join_price(window, price))
    row = list_grape_groups(general, min_bottles=1).row(0, named=True)
    slice_ = general.filter(
        (pl.col("wine_type") == row["wine_type"])
        & (pl.col("grape_majority") == row["grape_majority"])
    )
    section = build_catalog_section(
        row["wine_type"],
        row["grape_majority"],
        slice_,
        load_recommend_bundles(),
        settings.library_tables_dir,
        top=5,
    )
    assert section.n_priced == section.n_bottles
    # 25 USD snapshot, adjusted to ~2026 EUR, still clears the value cap.
    assert not section.best_value.is_empty()
    assert section.best_value.get_column("price_eur").to_list()[0] < VALUE_PRICE_CAP_EUR


def test_vintage_quality_one_row_per_type_vintage(trained_bundles: Settings) -> None:
    general = filter_library_candidates(join_price(filter_vintage_window(build_candidates()), None))
    vq = vintage_quality(general, load_recommend_bundles())
    assert vq.columns == ["wine_type", "vintage_year", "n_bottles", "mean_predicted_rating"]
    assert vq.height >= 1
    assert vq.select(["wine_type", "vintage_year"]).is_duplicated().sum() == 0
    assert vq.get_column("mean_predicted_rating").is_between(1.0, 5.5).all()
