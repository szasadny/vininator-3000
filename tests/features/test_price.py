"""Tests for features/price.py — the post-hoc price snapshot matcher.

All offline: a tiny synthetic Wine Reviews CSV plus a synthetic X-Wines wines
parquet under `tmp_data_dir`. No network, no real Kaggle download.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from vininator.config import Settings, get_settings
from vininator.features.price import (
    PRICE_SCHEMA,
    _norm,
    _tokens,
    build_price_table,
    load_price_source,
    match_prices,
    scan_price,
)

_WINES_SCHEMA: dict[str, pl.DataType] = {
    "WineID": pl.Int64(),
    "WineName": pl.String(),
    "WineryName": pl.String(),
    "RegionName": pl.String(),
    "Grapes": pl.List(pl.String()),
}


def _write_wines(settings: Settings, rows: list[dict]) -> None:
    pl.DataFrame(rows, schema=_WINES_SCHEMA).write_parquet(settings.xwines_wines_parquet)


def _write_reviews(settings: Settings, rows: list[dict]) -> None:
    csv_path = settings.wine_reviews_csv
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_csv(csv_path)


def _wines() -> list[dict]:
    return [
        {
            "WineID": 1,
            "WineName": "Pinot Gris",
            "WineryName": "Rainstorm",
            "RegionName": "Willamette Valley",
            "Grapes": ["Pinot Gris"],
        },
        {
            "WineID": 2,
            "WineName": "Merlot Reserve",
            "WineryName": "Rainstorm",
            "RegionName": "Willamette Valley",
            "Grapes": ["Merlot"],
        },
        {
            "WineID": 3,
            "WineName": "Gran Reserva",
            "WineryName": "Bodega Nadie",
            "RegionName": "Rioja",
            "Grapes": ["Tempranillo"],
        },
    ]


def _reviews() -> list[dict]:
    # Rainstorm: 2 Pinot Gris rows (exact for wine 1), 1 Merlot Reserve (exact for
    # wine 2), 3 rows total >= PRICE_MIN_WINERY_ROWS. Bodega Nadie: 2 rows, below
    # the winery-median floor and no name-token match -> wine 3 stays unmatched.
    return [
        {"winery": "Rainstorm", "title": "Rainstorm 2013 Pinot Gris (Willamette)", "price": 20.0},
        {"winery": "Rainstorm", "title": "Rainstorm 2014 Pinot Gris (Willamette)", "price": 24.0},
        {
            "winery": "Rainstorm",
            "title": "Rainstorm 2015 Merlot Reserve (Willamette)",
            "price": 30.0,
        },
        {"winery": "Bodega Nadie", "title": "Bodega Nadie 2016 Blanco (Rioja)", "price": 12.0},
        {"winery": "Bodega Nadie", "title": "Bodega Nadie 2017 Rosado (Rioja)", "price": 14.0},
    ]


def test_norm_folds_accents_and_punctuation() -> None:
    assert _norm("Château Léoville-Barton") == "chateau leoville barton"
    assert _norm(None) == ""
    assert _tokens("Vinha  Maria!! 2018") == ["vinha", "maria", "2018"]


def test_missing_csv_raises_with_path(tmp_data_dir: Path) -> None:
    settings = get_settings()
    with pytest.raises(FileNotFoundError) as exc:
        load_price_source()
    assert str(settings.wine_reviews_csv) in str(exc.value)


def test_exact_beats_winery_median(tmp_data_dir: Path) -> None:
    settings = get_settings()
    _write_wines(settings, _wines())
    _write_reviews(settings, _reviews())

    priced = build_price_table(force=True).sort("wine_id")
    assert priced.schema == PRICE_SCHEMA
    by_id = {r["wine_id"]: r for r in priced.iter_rows(named=True)}

    # Wine 1: exact, median of the two Pinot Gris rows (20, 24) = 22.
    assert by_id[1]["match_confidence"] == "exact"
    assert by_id[1]["price_estimate"] == 22.0
    # Wine 2: exact single Merlot Reserve row = 30.
    assert by_id[2]["match_confidence"] == "exact"
    assert by_id[2]["price_estimate"] == 30.0
    # Wine 3: Bodega Nadie has only 2 rows (< floor) and no name match -> unmatched.
    assert 3 not in by_id


def test_winery_median_fallback_when_name_unmatched(tmp_data_dir: Path) -> None:
    settings = get_settings()
    # A wine whose name matches no title, but the winery has >= 3 priced rows.
    _write_wines(
        settings,
        [
            {
                "WineID": 9,
                "WineName": "Mystery Cuvee",
                "WineryName": "Rainstorm",
                "RegionName": "Willamette Valley",
                "Grapes": ["Merlot"],
            },
        ],
    )
    _write_reviews(settings, _reviews())

    priced = build_price_table(force=True)
    row = priced.filter(pl.col("wine_id") == 9).to_dicts()[0]
    assert row["match_confidence"] == "winery-median"
    # median of Rainstorm prices (20, 24, 30) = 24.
    assert row["price_estimate"] == 24.0


def test_match_prices_pure_no_source_rows() -> None:
    wines = pl.DataFrame({"wine_id": [1], "norm_winery": ["x"], "wine_tokens": [["a"]]})
    source = pl.DataFrame(
        {"norm_winery": [], "title_tokens": [], "price": []},
        schema={"norm_winery": pl.String, "title_tokens": pl.List(pl.String), "price": pl.Float64},
    )
    out = match_prices(wines, source)
    assert out.is_empty()
    assert out.schema == PRICE_SCHEMA


def test_build_atomic_and_force_rebuild(tmp_data_dir: Path) -> None:
    settings = get_settings()
    _write_wines(settings, _wines())
    _write_reviews(settings, _reviews())

    build_price_table(force=True)
    assert settings.price_parquet.exists()
    # No stray temp file left behind by the atomic write.
    assert not settings.price_parquet.with_suffix(".parquet.tmp").exists()

    # Cache-first: a second call without force returns the same rows without error.
    cached = build_price_table(force=False)
    assert cached.height == scan_price().collect().height
