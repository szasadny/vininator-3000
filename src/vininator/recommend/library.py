"""Catalog logic for the recommendation library (scripts/build_library.py).

The library is the Phase 6 recommender applied per grape and per favorite wine,
restricted to recent vintages, decorated with post-hoc price/value metadata. All
model scoring goes through the existing shared path — `score_at_opening_year`,
`enrich_profile`, `recommend_age_well` — so a wine's numbers here are identical
to what the CLI recommenders produce. Nothing trains, nothing fetches terroir,
and price is joined (never fed to the model) after scoring.

Layering: price *sourcing* lives in `features/price.py`; this module only joins
the prebuilt `price.parquet`, converts USD → EUR for display, and builds the
value views. Everything is a pure function over frames except `build_catalog_
section`, which drives the recommenders.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import polars as pl

from vininator.config import (
    DEFAULT_OPENING_YEAR,
    LIBRARY_FAVORITE_GRAPES,
    LIBRARY_FAVORITE_REGIONS,
    LIBRARY_FUTURE_PEAK_MIN_YEAR,
    LIBRARY_MIN_BOTTLES_PER_GRAPE,
    LIBRARY_MIN_VINTAGE,
    LIBRARY_TOP_N,
    LIBRARY_VINTAGE_QUALITY_AGE,
    LIBRARY_WINE_TYPES,
    PRICE_APPRECIATION_ANNUAL,
    PRICE_APPRECIATION_MAX_YEARS,
    PRICE_BAND_TOP,
    PRICE_BANDS,
    PRICE_INFLATION_ANNUAL,
    PRICE_SNAPSHOT_YEAR,
    PRICE_USD_PER_EUR,
    RECOMMEND_HORIZON_YEARS,
    VALUE_PRICE_CAP_EUR,
)
from vininator.models.dataset import NotifyFn, notify
from vininator.recommend.age_well import recommend_age_well
from vininator.recommend.drink_now import (
    DRINK_NOW_OUT_COLS,
    Bundles,
    RecommendFilters,
    apply_filters,
    collapse_distinct_wines,
    enrich_profile,
    score_at_opening_year,
    train_age_bounds,
)

# Drink-now columns on the library pages: the shared set plus the price trio.
LIBRARY_DRINK_NOW_COLS: tuple[str, ...] = (
    *DRINK_NOW_OUT_COLS,
    "price_eur",
    "price_band",
    "match_confidence",
)


@dataclasses.dataclass(frozen=True)
class CatalogSection:
    """One page section: a grape (type pages) or a favorite wine/style.

    `label` is the heading — a grape name on the type pages, a wine-style display
    name on the favorites page (Amarone and Super Tuscan are styles defined by
    region, not grapes). Every table is already sliced to its top-N.
    """

    wine_type: str
    label: str
    n_bottles: int
    n_priced: int
    drink_now: pl.DataFrame
    age_well: pl.DataFrame
    future_greats: pl.DataFrame
    best_value: pl.DataFrame  # top-N under the euro cap, by predicted rating
    best_value_score: pl.DataFrame  # top-N by rating-per-euro


# ---------------------------------------------------------------------------
# Candidate filtering
# ---------------------------------------------------------------------------


def filter_vintage_window(candidates: pl.DataFrame) -> pl.DataFrame:
    """Keep only the library's recent window (`vintage_year > LIBRARY_MIN_VINTAGE`).

    The favorites and best-years slices start here — blends stay in, since the
    monogrape restriction only applies to the per-grape type pages.
    """
    return candidates.filter(pl.col("vintage_year") > LIBRARY_MIN_VINTAGE)


def filter_library_candidates(window: pl.DataFrame) -> pl.DataFrame:
    """The general per-grape pool: monogrape wines of the library's wine types."""
    return window.filter(
        pl.col("is_monogrape")
        & pl.col("wine_type").is_in(LIBRARY_WINE_TYPES)
        & pl.col("grape_majority").is_not_null()
    )


# ---------------------------------------------------------------------------
# Price join (post-hoc metadata; never a model feature)
# ---------------------------------------------------------------------------


def join_price(candidates: pl.DataFrame, price: pl.DataFrame | None) -> pl.DataFrame:
    """Left-join the price snapshot and derive `price_eur` + `price_band`.

    `price=None` (no snapshot on disk) still produces every price column, all
    null with `match_confidence="none"`, so downstream code takes one path
    whether or not prices exist. Source prices are a 2017 USD snapshot; `price_eur`
    is that price brought to the opening year (inflation + a per-vintage aging
    premium, see `_current_price_eur_expr`) and converted to EUR.
    """
    if price is None:
        out = candidates.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("price_estimate"),
            pl.lit(None, dtype=pl.String).alias("price_source"),
            pl.lit(None, dtype=pl.String).alias("price_currency"),
            pl.lit("none", dtype=pl.String).alias("match_confidence"),
        )
    else:
        out = candidates.join(price, on="wine_id", how="left").with_columns(
            pl.col("match_confidence").fill_null("none")
        )
    out = out.with_columns(
        _current_price_eur_expr(pl.col("price_estimate"), pl.col("vintage_year")).alias("price_eur")
    )
    return out.with_columns(_price_band_expr(pl.col("price_eur")).alias("price_band"))


def _current_price_eur_expr(price_usd: pl.Expr, vintage_year: pl.Expr) -> pl.Expr:
    """Bring a 2017 snapshot USD price to opening-year EUR (inflation + aging).

    Two factors compound onto the snapshot: uniform USD inflation from the
    snapshot year to the opening year, and a fine-wine aging premium over the
    bottle's age in the opening year (older vintages cost more). The aging
    exponent is clipped to `[0, PRICE_APPRECIATION_MAX_YEARS]` so ancient vintages
    don't extrapolate wildly. A null price stays null (unpriced wines).
    """
    inflation = (1.0 + PRICE_INFLATION_ANNUAL) ** (DEFAULT_OPENING_YEAR - PRICE_SNAPSHOT_YEAR)
    bottle_age = (pl.lit(DEFAULT_OPENING_YEAR) - vintage_year).clip(0, PRICE_APPRECIATION_MAX_YEARS)
    appreciation = pl.lit(1.0 + PRICE_APPRECIATION_ANNUAL).pow(bottle_age)
    return price_usd * inflation * appreciation / PRICE_USD_PER_EUR


def _price_band_expr(price_eur: pl.Expr) -> pl.Expr:
    """Chain the `PRICE_BANDS` cutoffs into a band label (null when unpriced)."""
    expr = pl.when(price_eur.is_null()).then(pl.lit(None, dtype=pl.String))
    for label, upper in PRICE_BANDS:
        expr = expr.when(price_eur < upper).then(pl.lit(label))
    return expr.otherwise(pl.lit(PRICE_BAND_TOP))


# ---------------------------------------------------------------------------
# Grouping + trajectory views
# ---------------------------------------------------------------------------


def list_grape_groups(
    filtered: pl.DataFrame, min_bottles: int = LIBRARY_MIN_BOTTLES_PER_GRAPE
) -> pl.DataFrame:
    """`(wine_type, grape_majority, n_bottles)` for grapes clearing `min_bottles`.

    Sorted by wine-type order (`LIBRARY_WINE_TYPES`), then bottle count desc, then
    grape name — a stable, readable page order.
    """
    grouped = (
        filtered.group_by(["wine_type", "grape_majority"])
        .agg(pl.len().alias("n_bottles"))
        .filter(pl.col("n_bottles") >= min_bottles)
    )
    return (
        grouped.with_columns(_type_rank_expr(pl.col("wine_type")).alias("_type_rank"))
        .sort(["_type_rank", "n_bottles", "grape_majority"], descending=[False, True, False])
        .drop("_type_rank")
    )


def future_greats(
    summary: pl.DataFrame, top: int, min_peak_year: int = LIBRARY_FUTURE_PEAK_MIN_YEAR
) -> pl.DataFrame:
    """Latest peakers: bottles predicted to peak in `min_peak_year` or later.

    The dataset has no post-2021 vintages, so "future greats" means young bottles
    still climbing — an interior/late peak, not a wine already past its best. One
    row per wine (best vintage), like the other library tables.
    """
    ranked = summary.filter(
        (pl.col("predicted_peak_year") >= min_peak_year) & (pl.col("trajectory") != "declining")
    ).sort(["predicted_peak_rating", "wine_id", "vintage_year"], descending=[True, False, False])
    return collapse_distinct_wines(ranked).head(top)


def value_views(
    scored: pl.DataFrame, top: int, cap_eur: float = VALUE_PRICE_CAP_EUR
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """`(best-under-cap, best-rating-per-euro)` from a scored+priced frame.

    Unpriced rows are excluded from both views — never imputed. The first view is
    the best-rated bottles at or under `cap_eur`; the second is the highest
    rating-per-euro. A `value_score` column backs the second view's sort.
    """
    priced = scored.filter(pl.col("price_eur").is_not_null())
    under_cap = (
        priced.filter(pl.col("price_eur") <= cap_eur)
        .sort(["predicted_rating", "wine_id", "vintage_year"], descending=[True, False, False])
        .head(top)
    )
    by_value = (
        priced.with_columns((pl.col("predicted_rating") / pl.col("price_eur")).alias("value_score"))
        .sort(["value_score", "wine_id", "vintage_year"], descending=[True, False, False])
        .head(top)
    )
    return under_cap, by_value


# ---------------------------------------------------------------------------
# Section + favorites + best-vintages builders
# ---------------------------------------------------------------------------


def build_catalog_section(
    wine_type: str,
    label: str,
    section: pl.DataFrame,
    bundles: Bundles,
    tables_dir: Path,
    top: int = LIBRARY_TOP_N,
    opening_year: int = DEFAULT_OPENING_YEAR,
    age_bounds: tuple[int, int] | None = None,
    notify_fn: NotifyFn | None = None,
) -> CatalogSection:
    """Score one slice and assemble its five ranking tables.

    The slice is scored once at `opening_year` (drink-now + value share that
    pass); age-well runs its own horizon sweep via `recommend_age_well`. Every
    table shows one row per wine (best vintage) — producer dominance otherwise
    fills a section with a single estate. The slice must already carry the joined
    price columns. `top` bounds every table.
    """
    age_bounds = age_bounds or train_age_bounds()
    filtered = apply_filters(section, RecommendFilters(monogrape=False), opening_year)
    n_bottles = filtered.height
    # Count exact matches — the prices trustworthy enough to back the value tables.
    n_priced = filtered.filter(pl.col("match_confidence") == "exact").height

    if filtered.is_empty():
        empty_dn = section.head(0).select(
            [c for c in LIBRARY_DRINK_NOW_COLS if c in section.columns]
        )
        return CatalogSection(
            wine_type,
            label,
            0,
            0,
            empty_dn,
            section.head(0),
            section.head(0),
            empty_dn,
            empty_dn,
        )

    scored = enrich_profile(
        score_at_opening_year(filtered, opening_year, bundles, age_bounds), bundles
    )
    distinct = collapse_distinct_wines(
        scored.sort(
            ["predicted_rating", "wine_id", "vintage_year"], descending=[True, False, False]
        )
    )
    drink_now = distinct.head(top).select(list(LIBRARY_DRINK_NOW_COLS))
    # Value rankings use exact price matches only: a winery-median estimate
    # underprices a winery's flagship (its icon is the expensive tail of the
    # winery's range), which would salt the value list with mispriced trophies.
    best_value, best_value_score = value_views(_exact_priced(distinct), top)

    slug = _slug(f"{wine_type}-{label}")
    aw = recommend_age_well(
        opening_year=opening_year,
        horizon=RECOMMEND_HORIZON_YEARS,
        filters=RecommendFilters(monogrape=False),
        top=top,
        distinct_wines=True,
        candidates=section,
        bundles=bundles,
        out_path=tables_dir / f"{slug}_age_well.parquet",
        summary_path=tables_dir / f"{slug}_age_well_summary.parquet",
    )
    notify(notify_fn, f"    {wine_type}/{label}: {n_bottles} bottles, {n_priced} priced")
    return CatalogSection(
        wine_type=wine_type,
        label=label,
        n_bottles=n_bottles,
        n_priced=n_priced,
        drink_now=drink_now,
        age_well=aw.table,
        future_greats=future_greats(aw.summary, top),
        best_value=best_value,
        best_value_score=best_value_score,
    )


def favorite_slices(window: pl.DataFrame) -> list[tuple[str, pl.DataFrame]]:
    """`(display_name, slice)` per favorite: grapes (monogrape) then styles (region).

    Grapes match `grape_majority` case-insensitively among monogrape wines. The
    styles (Amarone, Super Tuscan) are region-defined blends, so they match
    `region_name` with the monogrape filter off, Red only.
    """
    out: list[tuple[str, pl.DataFrame]] = []
    for grape in LIBRARY_FAVORITE_GRAPES:
        slice_ = window.filter(
            pl.col("is_monogrape") & (pl.col("grape_majority").str.to_lowercase() == grape.lower())
        )
        out.append((grape, slice_))
    for display, regions in LIBRARY_FAVORITE_REGIONS:
        slice_ = window.filter(
            (pl.col("wine_type") == "Red")
            & pl.col("region_name").str.to_lowercase().is_in(list(regions))
        )
        out.append((display, slice_))
    return out


def vintage_quality(
    filtered: pl.DataFrame,
    bundles: Bundles,
    age: int = LIBRARY_VINTAGE_QUALITY_AGE,
    age_bounds: tuple[int, int] | None = None,
) -> pl.DataFrame:
    """Per (wine_type, vintage): mean predicted rating scored at a fixed age.

    Comparing every vintage at the same age (`age` years post-vintage) removes
    the age effect, so the ranking reflects vintage quality rather than "older
    wines score differently". Returns a long frame sorted by type then vintage.
    """
    age_bounds = age_bounds or train_age_bounds()
    frames: list[pl.DataFrame] = []
    for vintage in sorted(filtered.get_column("vintage_year").unique().to_list()):
        slice_ = filtered.filter(pl.col("vintage_year") == vintage)
        if slice_.is_empty():
            continue
        scored = score_at_opening_year(slice_, vintage + age, bundles, age_bounds)
        frames.append(
            scored.group_by("wine_type")
            .agg(
                pl.len().alias("n_bottles"),
                pl.col("predicted_rating").mean().alias("mean_predicted_rating"),
            )
            .with_columns(pl.lit(vintage).alias("vintage_year"))
        )
    combined = pl.concat(frames)
    return (
        combined.with_columns(_type_rank_expr(pl.col("wine_type")).alias("_type_rank"))
        .sort(["_type_rank", "vintage_year"], descending=[False, False])
        .drop("_type_rank")
        .select("wine_type", "vintage_year", "n_bottles", "mean_predicted_rating")
    )


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _exact_priced(scored: pl.DataFrame) -> pl.DataFrame:
    """Rows with an exact per-wine price match — the only prices trustworthy for value."""
    return scored.filter(pl.col("match_confidence") == "exact")


def _type_rank_expr(wine_type: pl.Expr) -> pl.Expr:
    """Sort key mapping each wine type to its `LIBRARY_WINE_TYPES` position."""
    expr = pl.when(wine_type == LIBRARY_WINE_TYPES[0]).then(0)
    for rank, name in enumerate(LIBRARY_WINE_TYPES[1:], start=1):
        expr = expr.when(wine_type == name).then(rank)
    return expr.otherwise(len(LIBRARY_WINE_TYPES))


def _slug(text: str) -> str:
    """Filesystem-safe slug: lowercase, non-alphanumerics → single hyphen."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
