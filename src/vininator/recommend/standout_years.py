"""Standout-of-the-year ranking — one "what to open this year" shortlist per year.

This is the drink-now score run across a window of opening years and curated
into a per-year top-N. It reuses `drink_now`'s scoring path verbatim, so a
wine's rank in year Y here is identical to its standalone `recommend_drink_now`
rank at year Y — the shortlist is editorial curation over the same numbers, not
a second model.

The static filters (grape / region / monogrape) and the profile enrichment are
computed once; only the vintage/age filter and the scoring vary across years.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import polars as pl

from vininator.config import STANDOUT_TOP_N, STANDOUT_YEAR_RANGE, get_settings
from vininator.models.dataset import NotifyFn, notify
from vininator.recommend.drink_now import (
    DRINK_NOW_OUT_COLS,
    Bundles,
    RecommendFilters,
    apply_static_filters,
    apply_vintage_filters,
    build_candidates,
    enrich_profile,
    load_recommend_bundles,
    score_at_opening_year,
    train_age_bounds,
    write_ranking_parquet,
)


@dataclasses.dataclass(frozen=True)
class StandoutReport:
    """Summary returned by `recommend_standout_years`."""

    table: pl.DataFrame
    path: Path
    from_year: int
    to_year: int
    years_covered: list[int]


def recommend_standout_years(
    *,
    from_year: int = STANDOUT_YEAR_RANGE[0],
    to_year: int = STANDOUT_YEAR_RANGE[1],
    filters: RecommendFilters | None = None,
    top: int = STANDOUT_TOP_N,
    out_path: Path | None = None,
    candidates: pl.DataFrame | None = None,
    bundles: Bundles | None = None,
    notify_fn: NotifyFn | None = None,
) -> StandoutReport:
    """Top-`top` wines to open in each year of `[from_year, to_year]`.

    For each opening year, scores the (statically filtered) candidates at that
    year's age, takes the top-`top` by predicted rating, and tags them with the
    year and their within-year rank. The blocks are concatenated into one
    long-format table keyed `(opening_year, rank, ...)`.

    Pass the raw `candidates` frame and `bundles` to reuse them across calls (see
    `recommend_drink_now`); this call still applies its own static filters.
    """
    filters = filters or RecommendFilters()
    out_path = out_path or get_settings().recommendations_standout_years_parquet

    bundles = bundles or load_recommend_bundles()
    raw = candidates if candidates is not None else build_candidates(notify_fn)
    base = apply_static_filters(raw, filters)
    base = enrich_profile(base, bundles)
    age_bounds = train_age_bounds()

    blocks: list[pl.DataFrame] = []
    years_covered: list[int] = []
    for year in range(from_year, to_year + 1):
        cands = apply_vintage_filters(base, filters, year)
        if cands.is_empty():
            notify(notify_fn, f"... no candidates for {year}, skipping")
            continue
        notify(notify_fn, f"... scoring {cands.height:,} wines at {year}")
        scored = score_at_opening_year(cands, year, bundles, age_bounds)
        block = (
            scored.sort(
                ["predicted_rating", "wine_id", "vintage_year"], descending=[True, False, False]
            )
            .head(top)
            .select(DRINK_NOW_OUT_COLS)
            .with_columns(pl.lit(year).alias("opening_year"))
            .with_row_index("rank", offset=1)
        )
        blocks.append(block.select(["opening_year", "rank", *DRINK_NOW_OUT_COLS]))
        years_covered.append(year)

    if not blocks:
        raise ValueError(
            f"No wines match the filters in any year of {from_year}-{to_year}. "
            "Loosen --grape / --region / --max-vintage-age, or drop --monogrape."
        )

    table = pl.concat(blocks)
    write_ranking_parquet(table, out_path)
    notify(notify_fn, f"... wrote {table.height} rows across {len(years_covered)} years")
    return StandoutReport(
        table=table,
        path=out_path,
        from_year=from_year,
        to_year=to_year,
        years_covered=years_covered,
    )
