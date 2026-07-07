"""Age-well ranking — project each wine's rating forward over opening years.

Reuses `drink_now`'s candidate table and scoring path: a candidate is a
`(wine_id, vintage_year)` bottle, scored at every opening year in
`[opening_year, opening_year + horizon]`. Because vintage is fixed and only
`age_at_review` advances, the trajectory is that bottle aging in the cellar.

Two artifacts come out: a long-format sweep (one row per bottle × opening year)
and a per-bottle summary (peak year / rating, slope to peak, and a trajectory
label) that ranks the cellar candidates — the wines still rising or peaking late
within the horizon.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import polars as pl

from vininator.config import (
    AGE_WELL_PLATEAU_EPS,
    DEFAULT_OPENING_YEAR,
    RECOMMEND_HORIZON_YEARS,
    RECOMMEND_TOP_N,
    get_settings,
)
from vininator.models.dataset import NotifyFn, notify
from vininator.recommend.drink_now import (
    Bundles,
    RecommendFilters,
    apply_filters,
    build_candidates,
    collapse_distinct_wines,
    enrich_profile,
    load_recommend_bundles,
    score_at_opening_year,
    train_age_bounds,
    write_ranking_parquet,
)

# Long-format columns: identity + the opening year + prediction + clip flag.
_AGE_WELL_LONG_COLS: tuple[str, ...] = (
    "wine_id",
    "winery_name",
    "wine_name",
    "region_name",
    "vintage_year",
    "opening_year",
    "age_at_review",
    "predicted_rating",
    "predicted_rating_lo",
    "predicted_rating_hi",
    "age_clipped",
)


@dataclasses.dataclass(frozen=True)
class AgeWellReport:
    """Summary returned by `recommend_age_well`."""

    long: pl.DataFrame
    summary: pl.DataFrame
    table: pl.DataFrame
    long_path: Path
    summary_path: Path
    opening_year: int
    horizon: int


def recommend_age_well(
    *,
    opening_year: int = DEFAULT_OPENING_YEAR,
    horizon: int = RECOMMEND_HORIZON_YEARS,
    filters: RecommendFilters | None = None,
    top: int = RECOMMEND_TOP_N,
    distinct_wines: bool = False,
    out_path: Path | None = None,
    summary_path: Path | None = None,
    candidates: pl.DataFrame | None = None,
    bundles: Bundles | None = None,
    notify_fn: NotifyFn | None = None,
) -> AgeWellReport:
    """Sweep each candidate over `opening_year .. opening_year + horizon`.

    Candidates are filtered and profile-enriched once; the rating heads are then
    scored once per opening year (one pool per year keeps peak memory at a single
    candidate copy). The long sweep goes to `out_path`; the per-bottle summary —
    peak year/rating, slope to peak, trajectory — goes to `summary_path`. The
    returned `table` is the top-`top` non-declining bottles by peak rating.
    `distinct_wines=True` collapses that table to one row per wine (its best
    vintage) before the top-`top`, matching `recommend_drink_now`.

    Pass the raw `candidates` frame and `bundles` to reuse them across calls (see
    `recommend_drink_now`); this call still applies its own filters.
    """
    filters = filters or RecommendFilters()
    settings = get_settings()
    out_path = out_path or settings.recommendations_age_well_parquet
    summary_path = summary_path or settings.recommendations_age_well_summary_parquet

    bundles = bundles or load_recommend_bundles()
    raw = candidates if candidates is not None else build_candidates(notify_fn)
    candidates = apply_filters(raw, filters, opening_year)
    if candidates.is_empty():
        raise ValueError(
            f"No wines match the filters at opening year {opening_year}. "
            "Loosen --grape / --region / --max-vintage-age, or drop --monogrape."
        )
    candidates = enrich_profile(candidates, bundles)
    age_bounds = train_age_bounds()

    per_year: list[pl.DataFrame] = []
    for year in range(opening_year, opening_year + horizon + 1):
        notify(notify_fn, f"... scoring {candidates.height:,} wines at {year}")
        scored = score_at_opening_year(candidates, year, bundles, age_bounds)
        per_year.append(
            scored.with_columns(pl.lit(year).alias("opening_year")).select(_AGE_WELL_LONG_COLS)
        )
    long = pl.concat(per_year)

    summary = _summarize(long, opening_year=opening_year)
    ranked = summary.filter(pl.col("trajectory") != "declining").sort(
        ["predicted_peak_rating", "wine_id", "vintage_year"], descending=[True, False, False]
    )
    if distinct_wines:
        ranked = collapse_distinct_wines(ranked)
    table = ranked.head(top)

    write_ranking_parquet(long, out_path)
    write_ranking_parquet(summary, summary_path)
    notify(notify_fn, f"... wrote {long.height:,} sweep rows and {summary.height:,} summaries")
    return AgeWellReport(
        long=long,
        summary=summary,
        table=table,
        long_path=out_path,
        summary_path=summary_path,
        opening_year=opening_year,
        horizon=horizon,
    )


def _summarize(long: pl.DataFrame, *, opening_year: int) -> pl.DataFrame:
    """Collapse the long sweep to one row per bottle with trajectory features.

    `predicted_peak_year` breaks ties toward the earliest year that reaches the
    peak (open it as soon as it is at its best). `slope_to_peak` is the average
    per-year gain from the opening year to the peak. The trajectory label ranks
    cellar-worthiness: a flat line is `plateau`, a peak at the horizon end is
    `rising`, an interior peak is `peaks_late`, and a peak at the opening year
    (only decline after) is `declining` — the one class excluded from the table.
    """
    summary = long.group_by(["wine_id", "vintage_year"]).agg(
        pl.col("winery_name").first(),
        pl.col("wine_name").first(),
        pl.col("region_name").first(),
        pl.col("predicted_rating").sort_by("opening_year").first().alias("rating_at_opening"),
        pl.col("predicted_rating").max().alias("predicted_peak_rating"),
        pl.col("predicted_rating").min().alias("_trough_rating"),
        pl.col("opening_year")
        .sort_by(["predicted_rating", "opening_year"], descending=[True, False])
        .first()
        .alias("predicted_peak_year"),
        pl.col("opening_year").max().alias("_last_year"),
        pl.col("age_clipped").any().alias("age_clipped_any"),
    )
    return (
        summary.with_columns(
            (
                (pl.col("predicted_peak_rating") - pl.col("rating_at_opening"))
                / pl.max_horizontal(pl.col("predicted_peak_year") - opening_year, pl.lit(1))
            ).alias("slope_to_peak"),
            pl.when(
                pl.col("predicted_peak_rating") - pl.col("_trough_rating") <= AGE_WELL_PLATEAU_EPS
            )
            .then(pl.lit("plateau"))
            .when(pl.col("predicted_peak_year") == pl.col("_last_year"))
            .then(pl.lit("rising"))
            .when(pl.col("predicted_peak_year") > opening_year)
            .then(pl.lit("peaks_late"))
            .otherwise(pl.lit("declining"))
            .alias("trajectory"),
        )
        .drop("_trough_rating", "_last_year")
        .sort(["wine_id", "vintage_year"])
    )
