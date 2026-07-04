"""Join climate.parquet ⨝ soil.parquet → data/interim/terroir.parquet.

Phase 2 step 4. PRs 1-3 produced per-region soil/terrain rows and per
(region, vintage_year) climate rows; this module fuses them into the single
canonical block that downstream feature assembly (Phase 3 build.py) and the
live `TerroirProvider` (Phase 7) consume. No consumer reaches into
climate.parquet or soil.parquet directly once this lands.

The join is climate-driven: every climate row gets soil columns left-joined
onto it, broadcast across the 31 vintages a region has. Regions present in
soil but missing from climate are silently dropped — without a climate
signal we can't model wines from that region anyway. Climate rows whose
matching soil row is missing carry null soil columns; `soil_status` is null
on those rows, which downstream code can filter on if it needs an
all-features-present subset.

There is no `compute_terroir_*` pure helper. The row-level math reused at
inference time lives in `compute_climate_features` and `compute_soil_features`
in PR2/PR3; the joiner is a pure compose over already-materialised parquets.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import polars as pl

from vininator.config import get_settings
from vininator.features.climate import scan_climate
from vininator.features.soil import scan_soil

NotifyFn = Callable[[str], None]


TERROIR_SCHEMA: dict[str, pl.DataType] = {
    "region": pl.String(),
    "country": pl.String(),
    "vintage_year": pl.Int64(),
    "lat": pl.Float64(),
    "lon": pl.Float64(),
    "gdd_10c": pl.Float64(),
    "precip_total_mm": pl.Float64(),
    "precip_harvest_mm": pl.Float64(),
    "heat_spike_days": pl.Int64(),
    "frost_days_spring": pl.Int64(),
    "diurnal_range_mean": pl.Float64(),
    "solar_total_mj": pl.Float64(),
    "gdd_10c_anom": pl.Float64(),
    "precip_total_mm_anom": pl.Float64(),
    "precip_harvest_mm_anom": pl.Float64(),
    "heat_spike_days_anom": pl.Float64(),
    "frost_days_spring_anom": pl.Float64(),
    "diurnal_range_mean_anom": pl.Float64(),
    "solar_total_mj_anom": pl.Float64(),
    "is_partial": pl.Boolean(),
    "climate_status": pl.String(),
    "climate_error": pl.String(),
    "climate_fetched_at": pl.Datetime("us", "UTC"),
    "clay_pct": pl.Float64(),
    "sand_pct": pl.Float64(),
    "silt_pct": pl.Float64(),
    "ph_h2o": pl.Float64(),
    "soc_gkg": pl.Float64(),
    "cec_cmolkg": pl.Float64(),
    "bdod_kgdm3": pl.Float64(),
    "coarse_frag_pct": pl.Float64(),
    "elevation_m": pl.Float64(),
    "slope_deg": pl.Float64(),
    "drainage_class": pl.String(),
    "calcareous": pl.Boolean(),
    "soil_status": pl.String(),
    "soil_error": pl.String(),
    "soil_fetched_at": pl.Datetime("us", "UTC"),
}

# Columns that never become model features: QA rollup metadata and the
# geographic join keys (region/country/vintage_year are carried by the
# ratings side of the Phase 3 join; lat/lon are geometry, not terroir).
_NON_FEATURE_SUFFIXES: tuple[str, ...] = ("_status", "_error", "_fetched_at")
_NON_FEATURE_COLS: frozenset[str] = frozenset({"region", "country", "lat", "lon", "vintage_year"})


def terroir_feature_cols() -> list[str]:
    """The terroir columns that enter models as features, in schema order.

    Single source for "what is the terroir block" — Phase 3 assembly
    (features/build.py) carries exactly these columns into the processed
    table, and the Phase 5 ablation drops exactly these columns to measure
    the block's contribution. Deriving both from TERROIR_SCHEMA keeps them
    in lockstep when the schema grows.
    """
    return [
        c
        for c in TERROIR_SCHEMA
        if not any(c.endswith(s) for s in _NON_FEATURE_SUFFIXES) and c not in _NON_FEATURE_COLS
    ]


def build_terroir_table(
    *,
    force: bool = False,  # noqa: ARG001 — passthrough for CLI symmetry with features {soil,climate}.
    notify_fn: NotifyFn | None = None,
) -> Path:
    """Left-join soil onto climate on (region, country); write terroir.parquet.

    Each side's `status` / `error` / `fetched_at` is renamed with a
    `climate_*` / `soil_*` prefix so the rollup is unambiguous; soil's `lat`
    and `lon` are dropped since climate's (identical) pair already carries
    the geometry. The build always rebuilds from current inputs — the
    operation is a single polars join and there is no incremental state
    worth preserving.
    """
    settings = get_settings()
    settings.ensure_dirs()

    climate = scan_climate().rename(
        {
            "status": "climate_status",
            "error": "climate_error",
            "fetched_at": "climate_fetched_at",
        }
    )
    soil = scan_soil().drop("lat", "lon").rename(
        {
            "status": "soil_status",
            "error": "soil_error",
            "fetched_at": "soil_fetched_at",
        }
    )

    # nulls_equal=True so (region, country=None) rows match each other instead
    # of silently dropping. Current data has no null countries, but the schema
    # allows them and the join contract should be explicit either way.
    joined = climate.join(
        soil,
        on=["region", "country"],
        how="left",
        nulls_equal=True,
    ).collect()

    # Pin column order to TERROIR_SCHEMA so the schema test guards drift.
    out = joined.select(list(TERROIR_SCHEMA.keys()))

    target = settings.terroir_parquet
    _write_parquet_atomic(out, target)

    if notify_fn is not None:
        notify_fn(f"Wrote {out.height} terroir rows to {target}")
    return target


def scan_terroir() -> pl.LazyFrame:
    """Lazy frame over the terroir parquet."""
    path = get_settings().terroir_parquet
    if not path.exists():
        raise FileNotFoundError(
            f"Terroir parquet not found at {path}. "
            "Run `uv run vininator features terroir` first."
        )
    return pl.scan_parquet(path)


def _write_parquet_atomic(df: pl.DataFrame, target: Path) -> None:
    tmp = target.with_suffix(target.suffix + ".tmp")
    df.write_parquet(tmp)
    tmp.replace(target)
