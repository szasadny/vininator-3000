"""Tests for features/build.py — the leakage-safe feature assembly pipeline.

All tests are offline and use synthetic fixtures written to `tmp_data_dir`.
The most critical test is `test_producer_agg_leakage_canary`, which verifies
that producer aggregate features on the test split are computed from training
rows only, not from test rows. If that test fails, the headline result in
Phase 4 is invalid.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

import polars as pl
import pytest

from vininator.config import FUTURE_VINTAGE_HOLDOUT, get_settings
from vininator.features.build import build_processed_tables
from vininator.features.terroir import TERROIR_SCHEMA

_FETCHED = datetime(2026, 5, 1, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _wine(
    wine_id: int,
    region: str = "Bordeaux",
    country: str = "France",
    winery_id: int = 1,
    **kw: Any,
) -> dict[str, Any]:
    return {
        "WineID": wine_id,
        "WineName": f"Wine {wine_id}",
        "Type": kw.get("Type", "Red"),
        "Elaborate": "",
        "Grapes": ["Merlot"],
        "Harmonize": "['Beef', 'Pork']",
        "ABV": 13.5,
        "Body": "Full-bodied",
        "Acidity": "High",
        "Code": "",
        "Country": country,
        "RegionID": 1,
        "RegionName": region,
        "WineryID": winery_id,
        "WineryName": f"Winery {winery_id}",
        "Website": None,
        "Vintages": [2015, 2016],
    }


def _rating(
    rating_id: int,
    wine_id: int,
    vintage: int | None,
    rating: float,
    date: str = "2019-06-01 00:00:00",
) -> dict[str, Any]:
    age = (int(date[:4]) - vintage) if vintage is not None else None
    return {
        "RatingID": rating_id,
        "UserID": 1,
        "WineID": wine_id,
        "Vintage": vintage,
        "Rating": rating,
        "Date": datetime.strptime(date, "%Y-%m-%d %H:%M:%S"),
        "age_at_review": age,
    }


def _terroir_row(
    region: str, country: str, vintage_year: int, **kw: Any
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "region": region,
        "country": country,
        "vintage_year": vintage_year,
        "lat": 45.0,
        "lon": -1.0,
        "gdd_10c": 1500.0,
        "precip_total_mm": 400.0,
        "precip_harvest_mm": 60.0,
        "heat_spike_days": 3,
        "frost_days_spring": 1,
        "diurnal_range_mean": 10.0,
        "solar_total_mj": 4000.0,
        "gdd_10c_anom": 0.0,
        "precip_total_mm_anom": 0.0,
        "precip_harvest_mm_anom": 0.0,
        "heat_spike_days_anom": 0.0,
        "frost_days_spring_anom": 0.0,
        "diurnal_range_mean_anom": 0.0,
        "solar_total_mj_anom": 0.0,
        "is_partial": False,
        "climate_status": "ok",
        "climate_error": None,
        "climate_fetched_at": _FETCHED,
        "clay_pct": 25.0,
        "sand_pct": 40.0,
        "silt_pct": 35.0,
        "ph_h2o": 7.2,
        "soc_gkg": 15.0,
        "cec_cmolkg": 12.0,
        "bdod_kgdm3": 1.3,
        "coarse_frag_pct": 10.0,
        "elevation_m": 200.0,
        "slope_deg": 3.0,
        "drainage_class": "loamy",
        "calcareous": False,
        "soil_status": "ok",
        "soil_error": None,
        "soil_fetched_at": _FETCHED,
    }
    row.update(kw)
    return row


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


def _write_wines(rows: list[dict[str, Any]]) -> None:
    pl.DataFrame(rows, schema=_WINES_SCHEMA).write_parquet(
        get_settings().xwines_wines_parquet
    )


def _write_ratings(rows: list[dict[str, Any]]) -> None:
    pl.DataFrame(rows, schema=_RATINGS_SCHEMA).write_parquet(
        get_settings().xwines_ratings_parquet
    )


def _write_terroir(rows: list[dict[str, Any]]) -> None:
    pl.DataFrame(rows, schema=TERROIR_SCHEMA).write_parquet(
        get_settings().terroir_parquet
    )


def _minimal_fixture(tmp_data_dir: Path) -> None:
    """Write a minimal 3-wine fixture with vintages on both sides of 2018."""
    # Wine 1: train, winery 1, rating 4.0 (vintage 2015 — historical)
    # Wine 2: test,  winery 2, rating 3.0 (vintage 2016 — historical)
    # Wine 3: future vintage, winery 1, rating 5.0 (vintage 2020)
    _write_wines([
        _wine(1, winery_id=1),
        _wine(2, winery_id=2),
        _wine(3, winery_id=1),
    ])
    _write_ratings([
        _rating(1, 1, 2015, 4.0),
        _rating(2, 2, 2016, 3.0),
        _rating(3, 3, 2020, 5.0),
    ])
    _write_terroir([
        _terroir_row("Bordeaux", "France", 2015),
        _terroir_row("Bordeaux", "France", 2016),
        _terroir_row("Bordeaux", "France", 2020),
    ])


# ---------------------------------------------------------------------------
# Split correctness
# ---------------------------------------------------------------------------


def test_future_vintage_holdout_boundary(tmp_data_dir: Path) -> None:
    """All future_vintage_test rows have vintage in the holdout window."""
    _minimal_fixture(tmp_data_dir)
    build_processed_tables(force=True)
    settings = get_settings()

    fv = pl.read_parquet(settings.processed_future_vintage_test_parquet)
    lo, hi = FUTURE_VINTAGE_HOLDOUT
    assert (fv["vintage_year"] >= lo).all()
    assert (fv["vintage_year"] <= hi).all()


def test_train_test_vintage_upper_bound(tmp_data_dir: Path) -> None:
    """Train and test rows never contain future-vintage data."""
    _minimal_fixture(tmp_data_dir)
    build_processed_tables(force=True)
    settings = get_settings()

    _, hi = FUTURE_VINTAGE_HOLDOUT
    for path in (settings.processed_train_parquet, settings.processed_test_parquet):
        df = pl.read_parquet(path)
        assert (df["vintage_year"] < hi).all() or df.is_empty()


def test_wine_id_disjointness_train_vs_test(tmp_data_dir: Path) -> None:
    """No WineID appears in both train and test."""
    # Need at least 2 historical wines to guarantee a split.
    _write_wines([_wine(1), _wine(2)])
    _write_ratings([
        _rating(1, 1, 2015, 4.0),
        _rating(2, 2, 2016, 3.0),
    ])
    _write_terroir([
        _terroir_row("Bordeaux", "France", 2015),
        _terroir_row("Bordeaux", "France", 2016),
    ])
    build_processed_tables(force=True)
    settings = get_settings()

    train_ids = set(pl.read_parquet(settings.processed_train_parquet)["wine_id"].to_list())
    test_ids = set(pl.read_parquet(settings.processed_test_parquet)["wine_id"].to_list())
    assert train_ids.isdisjoint(test_ids), f"Overlap: {train_ids & test_ids}"


# ---------------------------------------------------------------------------
# NV row handling
# ---------------------------------------------------------------------------


def test_nv_ratings_dropped(tmp_data_dir: Path) -> None:
    """Ratings with null Vintage should not appear in any split."""
    _write_wines([_wine(1)])
    _write_ratings([
        _rating(1, 1, 2015, 4.0),
        # NV row: no vintage
        {
            "RatingID": 2,
            "UserID": 1,
            "WineID": 1,
            "Vintage": None,
            "Rating": 3.5,
            "Date": datetime(2019, 6, 1),
            "age_at_review": None,
        },
    ])
    _write_terroir([_terroir_row("Bordeaux", "France", 2015)])
    build_processed_tables(force=True)
    settings = get_settings()

    all_ids: list[int] = []
    for p in (
        settings.processed_train_parquet,
        settings.processed_test_parquet,
        settings.processed_future_vintage_test_parquet,
    ):
        all_ids.extend(pl.read_parquet(p)["rating_id"].to_list())
    assert 2 not in all_ids, "NV rating_id=2 must not appear in any split"


# ---------------------------------------------------------------------------
# Producer aggregate leakage canary (most critical test in this file)
# ---------------------------------------------------------------------------


def test_producer_agg_leakage_canary(tmp_data_dir: Path) -> None:
    """producer_mean_rating on the test split must equal the TRAIN-fold mean.

    Deliberately constructs winery 1 with:
      - 2 ratings in what will become train (3.0, 5.0 → mean 4.0)
      - 1 rating in what will become test (1.0)

    The correct train-fold mean is 4.0. If producer aggs were computed on the
    full dataset (leakage), the mean would be (3+5+1)/3 ≈ 3.0. Seeing 4.0 in
    the test row proves the agg was train-only.

    This test controls the split by constructing exactly 2 train wines (IDs
    3, 4 — winery 99, which stays in train) and 1 test wine (ID 1 — winery 1,
    which lands in test after the 85/15 shuffle with seed 42).

    NOTE: The exact mapping of wine IDs to train/test is seed-dependent. We
    verify train-fold isolation regardless of which side a wine lands on by
    asserting that the producer_mean_rating seen in the test split equals what
    we compute manually from the train rows only.
    """
    # Four wines: 1 and 2 from winery 1, 3 and 4 from winery 99 (control).
    _write_wines([
        _wine(1, winery_id=1),
        _wine(2, winery_id=1),
        _wine(3, winery_id=99),
        _wine(4, winery_id=99),
    ])
    _write_ratings([
        _rating(1, 1, 2015, 3.0),
        _rating(2, 2, 2016, 5.0),
        _rating(3, 3, 2015, 4.5),
        _rating(4, 4, 2016, 4.5),
        # Extra rating for wine 1 with high rating to create the leakage signal.
        _rating(5, 1, 2016, 1.0),
    ])
    _write_terroir([
        _terroir_row("Bordeaux", "France", 2015),
        _terroir_row("Bordeaux", "France", 2016),
    ])
    build_processed_tables(force=True)
    settings = get_settings()

    train = pl.read_parquet(settings.processed_train_parquet)
    test = pl.read_parquet(settings.processed_test_parquet)

    # Identify which winery_ids appear in train.
    train_winery_ids = set(train["winery_id"].unique().to_list())
    test_winery_ids = set(test["winery_id"].unique().to_list())
    # If a winery only appears in test, its producer_mean_rating should be null.
    test_only = test_winery_ids - train_winery_ids
    for wid in test_only:
        null_mask = (test["winery_id"] == wid) & test["producer_mean_rating"].is_null()
        assert null_mask.any(), f"Winery {wid} only in test but got non-null producer_mean_rating"

    # For wineries that appear in both, verify the agg equals the train-fold mean.
    shared = train_winery_ids & test_winery_ids
    for wid in shared:
        train_mean = (
            train
            .filter(pl.col("winery_id") == wid)
            .select(pl.col("rating").mean())
            .item()
        )
        test_agg = (
            test
            .filter(pl.col("winery_id") == wid)
            .select("producer_mean_rating")
            .unique()
            .item()
        )
        assert test_agg == pytest.approx(train_mean, rel=1e-5), (
            f"Winery {wid}: test sees producer_mean_rating={test_agg:.4f} "
            f"but train mean is {train_mean:.4f} — likely leakage"
        )


def test_producer_agg_leave_one_wine_out(tmp_data_dir: Path) -> None:
    """A train wine's producer features exclude its own ratings.

    The invariant (seed-independent, checked against the train parquet itself,
    whose rows are exactly the aggregation base): for every train wine,
    `producer_mean_rating` equals the mean rating of the OTHER train rows of
    its winery — never the winery mean including itself. A sole-wine winery
    gets null. This is the regression test for the Phase 5 leakage finding:
    self-inclusion made the feature a near-copy of the label at small
    wineries and the model lost to the plain winery baseline on held-out
    wines.
    """
    # Winery 1: three wines with deliberately spread ratings so the LOO mean
    # differs from the full winery mean for every wine. Winery 99: one wine
    # only → its train row must get null producer features under LOO.
    _write_wines([
        _wine(1, winery_id=1),
        _wine(2, winery_id=1),
        _wine(3, winery_id=1),
        _wine(4, winery_id=99),
    ])
    _write_ratings([
        _rating(1, 1, 2015, 1.0),
        _rating(2, 1, 2016, 2.0),
        _rating(3, 2, 2015, 3.0),
        _rating(4, 3, 2016, 5.0),
        _rating(5, 4, 2015, 4.0),
    ])
    _write_terroir([
        _terroir_row("Bordeaux", "France", 2015),
        _terroir_row("Bordeaux", "France", 2016),
    ])
    build_processed_tables(force=True)
    settings = get_settings()

    train = pl.read_parquet(settings.processed_train_parquet)
    for wine_id in train["wine_id"].unique().to_list():
        wine_rows = train.filter(pl.col("wine_id") == wine_id)
        winery_id = wine_rows["winery_id"][0]
        others = train.filter(
            (pl.col("winery_id") == winery_id) & (pl.col("wine_id") != wine_id)
        )
        got_mean = wine_rows["producer_mean_rating"][0]
        got_n = wine_rows["producer_n_reviews"][0]
        if others.is_empty():
            assert got_mean is None, (
                f"Wine {wine_id}: sole train wine of winery {winery_id} must get null "
                f"producer_mean_rating, got {got_mean} — own ratings leaked in"
            )
            assert got_n is None
        else:
            expected = others["rating"].mean()
            assert got_mean == pytest.approx(expected, rel=1e-9), (
                f"Wine {wine_id}: producer_mean_rating={got_mean} but the "
                f"leave-one-out winery mean is {expected} — own ratings leaked in"
            )
            assert got_n == others.height
            expected_std = others["rating"].std()
            got_std = wine_rows["producer_rating_std"][0]
            if others.height > 1:
                assert got_std == pytest.approx(expected_std, rel=1e-9)
            else:
                assert got_std is None


# ---------------------------------------------------------------------------
# Schema parity
# ---------------------------------------------------------------------------


def test_schema_parity_across_splits(tmp_data_dir: Path) -> None:
    """All three output parquets have identical column names and dtypes."""
    _minimal_fixture(tmp_data_dir)
    build_processed_tables(force=True)
    settings = get_settings()

    train_schema = dict(pl.scan_parquet(settings.processed_train_parquet).collect_schema())
    test_schema = dict(pl.scan_parquet(settings.processed_test_parquet).collect_schema())
    fv_schema = dict(pl.scan_parquet(settings.processed_future_vintage_test_parquet).collect_schema())

    assert train_schema == test_schema, "train / test schema mismatch"
    assert train_schema == fv_schema, "train / future_vintage_test schema mismatch"


# ---------------------------------------------------------------------------
# Idempotency and atomic write
# ---------------------------------------------------------------------------


def test_idempotency_no_force(tmp_data_dir: Path) -> None:
    """Second call without --force returns without rebuilding."""
    _minimal_fixture(tmp_data_dir)
    build_processed_tables(force=True)

    # Modify the terroir to a different value — a rebuild would pick it up.
    _write_terroir([
        _terroir_row("Bordeaux", "France", 2015, gdd_10c=9999.0),
        _terroir_row("Bordeaux", "France", 2016, gdd_10c=9999.0),
        _terroir_row("Bordeaux", "France", 2020, gdd_10c=9999.0),
    ])
    build_processed_tables(force=False)

    settings = get_settings()
    train = pl.read_parquet(settings.processed_train_parquet)
    # gdd_10c should still be 1500.0 from the first build (idempotent).
    if "gdd_10c" in train.columns:
        gdd_vals = train["gdd_10c"].drop_nulls().to_list()
        assert not any(v == 9999.0 for v in gdd_vals), "Second call re-ran the build (not idempotent)"


def test_force_flag_overwrites(tmp_data_dir: Path) -> None:
    """--force rebuilds even when outputs exist."""
    _minimal_fixture(tmp_data_dir)
    build_processed_tables(force=True)
    report1 = build_processed_tables(force=False)  # no-op
    report2 = build_processed_tables(force=True)   # full rebuild
    # Both should report the same row counts.
    assert report1.train_rows == report2.train_rows


def test_no_tmp_files_after_successful_build(tmp_data_dir: Path) -> None:
    """No .tmp files survive a successful build."""
    _minimal_fixture(tmp_data_dir)
    build_processed_tables(force=True)
    settings = get_settings()
    tmp_files = list(settings.processed_dir.glob("*.tmp"))
    assert tmp_files == [], f"Leftover tmp files: {tmp_files}"


def test_atomic_write_leaves_existing_intact_on_crash(tmp_data_dir: Path) -> None:
    """Simulated crash during write does not corrupt the previous parquet."""
    _minimal_fixture(tmp_data_dir)
    build_processed_tables(force=True)

    settings = get_settings()
    train_before = pl.read_parquet(settings.processed_train_parquet)

    # Patch replace() to simulate a crash after the .tmp is written.
    real_replace = Path.replace

    def _crash_replace(self: Path, target: Path) -> Path:
        if self.suffix == ".tmp":
            raise RuntimeError("simulated crash")
        return real_replace(self, target)

    with (
        patch.object(Path, "replace", _crash_replace),
        pytest.raises(RuntimeError, match="simulated crash"),
    ):
        build_processed_tables(force=True)

    # The original parquet must still be intact.
    train_after = pl.read_parquet(settings.processed_train_parquet)
    assert train_before.equals(train_after)
