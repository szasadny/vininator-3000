"""Overperformer outliers — wines the model rates well above their peer group.

For each wine at its drink-now opening year, compare the predicted rating to a
leakage-safe peer baseline: the per-`(grape_majority, region_name)` and
per-`(region_name, vintage_year)` mean rating, computed on the **train fold
only** (the same no-leakage rule as `eval/metrics.fit_rating_baselines`; we
don't reuse that fit because it carries no per-group support counts and needs
the eager train frame). A peer group counts only when it has enough distinct
training wines to be a stable baseline.

`overperformance = predicted_rating − peer_baseline`, and a wine survives only
when the lower confidence bound still clears the baseline — so a wide prediction
interval can't manufacture a fake outlier. The result is the "punching above
their weight" shortlist, deliberately not the highest absolute ratings.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import polars as pl

from vininator.config import (
    DEFAULT_OPENING_YEAR,
    OUTLIER_MIN_PEER_WINES,
    OUTLIER_PEER_AGG,
    RATING_TARGET,
    RECOMMEND_TOP_N,
    get_settings,
)
from vininator.models.dataset import NotifyFn, notify, split_path
from vininator.recommend.drink_now import (
    RecommendFilters,
    apply_filters,
    build_candidates,
    load_recommend_bundles,
    score_at_opening_year,
    train_age_bounds,
    write_ranking_parquet,
)

_OUTLIER_OUT_COLS: tuple[str, ...] = (
    "wine_id",
    "winery_name",
    "wine_name",
    "region_name",
    "vintage_year",
    "predicted_rating",
    "peer_baseline",
    "overperformance",
    "predicted_rating_lo",
    "predicted_rating_hi",
    "split",
)


@dataclasses.dataclass(frozen=True)
class OutlierReport:
    """Summary returned by `recommend_outliers`."""

    table: pl.DataFrame
    path: Path
    opening_year: int
    n_outliers: int


def recommend_outliers(
    *,
    opening_year: int = DEFAULT_OPENING_YEAR,
    filters: RecommendFilters | None = None,
    top: int = RECOMMEND_TOP_N,
    peer_agg: str = OUTLIER_PEER_AGG,
    min_peer_wines: int = OUTLIER_MIN_PEER_WINES,
    out_path: Path | None = None,
    notify_fn: NotifyFn | None = None,
) -> OutlierReport:
    """Rank wines by how far their predicted rating clears their peer baseline.

    Scores candidates at `opening_year`, attaches the two supported peer
    baselines, aggregates them (`peer_agg` = max by default — the stricter bar),
    keeps only wines whose `predicted_rating_lo` exceeds the baseline, and sorts
    by overperformance. All surviving outliers are written to `out_path`; the
    returned `table` is the top-`top` for the CLI.
    """
    filters = filters or RecommendFilters()
    out_path = out_path or get_settings().recommendations_outliers_parquet

    bundles = load_recommend_bundles()
    candidates = apply_filters(build_candidates(notify_fn), filters, opening_year)
    if candidates.is_empty():
        raise ValueError(
            f"No wines match the filters at opening year {opening_year}. "
            "Loosen --grape / --region / --max-vintage-age, or drop --monogrape."
        )
    age_bounds = train_age_bounds()
    scored = score_at_opening_year(candidates, opening_year, bundles, age_bounds)

    notify(notify_fn, "... attaching train-fold peer baselines")
    grape_region = _peer_lookup(["grape_majority", "region_name"], min_wines=min_peer_wines).rename(
        {"_peer_mean": "_peer_gr"}
    )
    region_vintage = _peer_lookup(["region_name", "vintage_year"], min_wines=min_peer_wines).rename(
        {"_peer_mean": "_peer_rv"}
    )
    scored = scored.join(grape_region, on=["grape_majority", "region_name"], how="left").join(
        region_vintage, on=["region_name", "vintage_year"], how="left"
    )

    agg = pl.max_horizontal if peer_agg == "max" else pl.mean_horizontal
    scored = scored.with_columns(agg("_peer_gr", "_peer_rv").alias("peer_baseline"))
    outliers = (
        scored.filter(pl.col("peer_baseline").is_not_null())
        .with_columns(
            (pl.col("predicted_rating") - pl.col("peer_baseline")).alias("overperformance")
        )
        .filter(pl.col("predicted_rating_lo") > pl.col("peer_baseline"))
        .sort(["overperformance", "wine_id", "vintage_year"], descending=[True, False, False])
        .select(_OUTLIER_OUT_COLS)
    )

    write_ranking_parquet(outliers, out_path)
    notify(notify_fn, f"... {outliers.height} outliers cleared the baseline gate")
    return OutlierReport(
        table=outliers.head(top),
        path=out_path,
        opening_year=opening_year,
        n_outliers=outliers.height,
    )


def _peer_lookup(keys: list[str], *, min_wines: int = OUTLIER_MIN_PEER_WINES) -> pl.DataFrame:
    """Train-fold mean rating per peer group, gated by distinct-wine support.

    Read straight off `train.parquet` (never the eval frames) so the baseline is
    leakage-safe under both the wine split and the future-vintage split. Groups
    with fewer than `min_wines` distinct wines are dropped — a per-group mean over
    one or two wines is too noisy to call an overperformance baseline.
    """
    return (
        pl.scan_parquet(split_path("train"))
        .group_by(keys)
        .agg(
            pl.col(RATING_TARGET).mean().alias("_peer_mean"),
            pl.col("wine_id").n_unique().alias("_peer_wines"),
        )
        .filter(pl.col("_peer_wines") >= min_wines)
        .drop("_peer_wines")
        .collect()
    )
