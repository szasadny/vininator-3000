"""Shared fixtures for the Phase 4 model tests.

`processed_dataset` writes a small synthetic X-Wines + terroir set to the
`tmp_data_dir` and runs the real `build_processed_tables`, so the model
trainers exercise the same parquets they will in production — just tiny. The
data is deliberately varied (multiple regions, wineries, grapes, body/acidity
classes, food pairings, and both historical and future vintages) so every
split is non-empty and the classifiers see more than one class.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl
import pytest
import yaml

from vininator.config import get_settings
from vininator.features.build import build_processed_tables
from vininator.features.terroir import TERROIR_SCHEMA

_REGIONS = [("Bordeaux", "France"), ("Rioja", "Spain"), ("Mosel", "Germany")]
_GRAPES = [["Merlot"], ["Cabernet Sauvignon"], ["Pinot Noir"], ["Riesling"], ["Tempranillo"]]
_HARMONIZE = [
    "['Beef', 'Pork']",
    "['Shellfish', 'Lean Fish']",
    "['Poultry', 'Veal']",
    "['Beef', 'Maturated Cheese']",
    "['Poultry', 'Shellfish']",
]
_BODIES = ["Full-bodied", "Medium-bodied", "Light-bodied"]
_ACIDITIES = ["High", "Medium", "Low"]

_WINES_SCHEMA: dict[str, pl.DataType] = {
    "WineID": pl.Int64(),
    "WineName": pl.String(),
    "Type": pl.String(),
    "Elaborate": pl.String(),
    "Grapes": pl.List(pl.String()),
    "Harmonize": pl.String(),
    "ABV": pl.Float64(),
    "Body": pl.String(),
    "Acidity": pl.String(),
    "Code": pl.String(),
    "Country": pl.String(),
    "RegionID": pl.Int64(),
    "RegionName": pl.String(),
    "WineryID": pl.Int64(),
    "WineryName": pl.String(),
    "Website": pl.String(),
    "Vintages": pl.List(pl.Int64()),
}

_RATINGS_SCHEMA: dict[str, pl.DataType] = {
    "RatingID": pl.Int64(),
    "UserID": pl.Int64(),
    "WineID": pl.Int64(),
    "Vintage": pl.Int64(),
    "Rating": pl.Float64(),
    "Date": pl.Datetime("us", None),
    "age_at_review": pl.Int64(),
}


def _terroir_row(region: str, country: str, vintage_year: int) -> dict[str, Any]:
    """A complete terroir row; numeric values vary by vintage to add signal."""
    row: dict[str, Any] = dict.fromkeys(TERROIR_SCHEMA, None)
    row.update(
        region=region,
        country=country,
        vintage_year=vintage_year,
        lat=45.0,
        lon=1.0,
        gdd_10c=1400.0 + 20.0 * (vintage_year - 2015),
        precip_total_mm=400.0,
        precip_harvest_mm=60.0,
        heat_spike_days=3,
        frost_days_spring=1,
        diurnal_range_mean=10.0,
        solar_total_mj=4000.0,
        gdd_10c_anom=0.0,
        precip_total_mm_anom=0.0,
        precip_harvest_mm_anom=0.0,
        heat_spike_days_anom=0.0,
        frost_days_spring_anom=0.0,
        diurnal_range_mean_anom=0.0,
        solar_total_mj_anom=0.0,
        is_partial=False,
        climate_status="ok",
        clay_pct=25.0,
        sand_pct=40.0,
        silt_pct=35.0,
        ph_h2o=7.2,
        soc_gkg=15.0,
        cec_cmolkg=12.0,
        bdod_kgdm3=1.3,
        coarse_frag_pct=10.0,
        elevation_m=200.0,
        slope_deg=3.0,
        drainage_class="loamy",
        calcareous=False,
        soil_status="ok",
    )
    return row


def _build_synthetic(n_wines: int = 48) -> None:
    """Write synthetic wines/ratings/terroir and run the real build pipeline."""
    settings = get_settings()
    wines: list[dict[str, Any]] = []
    ratings: list[dict[str, Any]] = []
    terroir_keys: set[tuple[str, int]] = set()
    rid = 1
    for w in range(1, n_wines + 1):
        region, country = _REGIONS[w % len(_REGIONS)]
        winery = (w % 5) + 1
        base = 3.0 + 0.5 * (winery % 4)  # producer signal: {3.0, 3.5, 4.0, 4.5}
        wines.append(
            {
                "WineID": w,
                "WineName": f"Wine {w}",
                "Type": "Red",
                "Elaborate": "",
                "Grapes": _GRAPES[w % len(_GRAPES)],
                "Harmonize": _HARMONIZE[w % len(_HARMONIZE)],
                "ABV": 13.0 + (w % 3),
                "Body": _BODIES[w % len(_BODIES)],
                "Acidity": _ACIDITIES[w % len(_ACIDITIES)],
                "Code": "",
                "Country": country,
                "RegionID": w % len(_REGIONS),
                "RegionName": region,
                "WineryID": winery,
                "WineryName": f"Winery {winery}",
                "Website": None,
                "Vintages": [2015, 2016, 2017, 2020],
            }
        )
        for v in (2015, 2016, 2017):
            jitter = 0.5 * (rid % 2)  # within-group spread for the quantile heads
            ratings.append(
                {
                    "RatingID": rid,
                    "UserID": 1,
                    "WineID": w,
                    "Vintage": v,
                    "Rating": min(5.0, base + jitter),
                    "Date": datetime(v + 2, 6, 1),
                    "age_at_review": 2,
                }
            )
            terroir_keys.add((region, v))
            rid += 1
        if w % 3 == 0:  # a third of wines get a future-vintage rating
            ratings.append(
                {
                    "RatingID": rid,
                    "UserID": 1,
                    "WineID": w,
                    "Vintage": 2020,
                    "Rating": base,
                    "Date": datetime(2021, 6, 1),
                    "age_at_review": 1,
                }
            )
            terroir_keys.add((region, 2020))
            rid += 1

    pl.DataFrame(wines, schema=_WINES_SCHEMA).write_parquet(settings.xwines_wines_parquet)
    pl.DataFrame(ratings, schema=_RATINGS_SCHEMA).write_parquet(settings.xwines_ratings_parquet)
    terroir_rows = [
        _terroir_row(region, country, v)
        for region, country in _REGIONS
        for (r, v) in terroir_keys
        if r == region
    ]
    pl.DataFrame(terroir_rows, schema=TERROIR_SCHEMA).write_parquet(settings.terroir_parquet)
    build_processed_tables(force=True)


@pytest.fixture
def processed_dataset(tmp_data_dir: Path):  # noqa: ARG001 — fixture wires settings to tmp
    """Build the synthetic processed parquets; yield the settings singleton."""
    _build_synthetic()
    return get_settings()


@pytest.fixture
def write_config(tmp_path: Path):
    """Factory writing a tiny experiment yaml (fast CatBoost) and returning its path."""

    def _make(name: str = "test", **catboost_overrides: Any) -> Path:
        params = {"iterations": 10, "depth": 3, "learning_rate": 0.3}
        params.update(catboost_overrides)
        path = tmp_path / f"{name}.yaml"
        path.write_text(
            yaml.safe_dump({"name": name, "catboost_params": params, "data": {"seed": 42}}),
            encoding="utf-8",
        )
        return path

    return _make
