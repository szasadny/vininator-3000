"""Drink-now ranking + the shared scoring path for every Phase 6 ranking.

This module owns three things the other recommenders reuse:

- `build_candidates` — the wine-vintage table scored by every ranking, gathered
  once from the processed splits (never re-engineered);
- `score_at_opening_year` — set `age_at_review = opening_year − vintage_year`,
  clip it to the trained age range, and predict the rating + quantile bands;
- `enrich_profile` — the age-independent body / acidity / top-pairings columns
  that decorate the ranking tables.

`age_well.py`, `standout_years.py`, and `outliers.py` import these so a wine's
predicted rating at a given opening year is identical no matter which ranking
surfaces it. Nothing here trains, and nothing fetches terroir: vintage is fixed,
so the terroir features are already on disk (the recommender rule in CLAUDE.md).
"""

from __future__ import annotations

import dataclasses
import gc
from pathlib import Path

import numpy as np
import polars as pl

from vininator.config import (
    ACIDITY_BUNDLE,
    AGE_COL,
    BODY_BUNDLE,
    DEFAULT_OPENING_YEAR,
    HARMONIZE_BUNDLE,
    HARMONIZE_TARGET_PREFIX,
    RATING_BUNDLE,
    RATING_QUANTILE_HI_BUNDLE,
    RATING_QUANTILE_LO_BUNDLE,
    RATING_TARGET,
    RECOMMEND_TOP_N,
    RECOMMEND_TOP_PAIRINGS,
    get_settings,
)
from vininator.models.artifacts import ModelBundle, load_bundle
from vininator.models.dataset import (
    NotifyFn,
    aggregate_wine_vintage,
    build_pool,
    load_split,
    notify,
    split_path,
)

# The output columns shared by drink-now and standout-of-the-year rankings, in
# report order. Identity + prediction + provenance (`split`, `age_clipped`,
# `n_ratings_cell`, `observed_mean_rating`) so the tables are self-documenting.
DRINK_NOW_OUT_COLS: tuple[str, ...] = (
    "wine_id",
    "winery_name",
    "wine_name",
    "region_name",
    "vintage_year",
    "age_at_review",
    "predicted_rating",
    "predicted_rating_lo",
    "predicted_rating_hi",
    "predicted_body",
    "predicted_acidity",
    "top_pairings",
    "age_clipped",
    "split",
    "n_ratings_cell",
    "observed_mean_rating",
)


@dataclasses.dataclass(frozen=True)
class RecommendFilters:
    """Candidate filters shared by every ranking command.

    `grape` / `region` are CLI slugs (`"pinot-noir"`) matched case-insensitively
    against `grape_majority` / `region_name`. `max_vintage_age` drops wines older
    than N years at the opening year (the "fresh-style" cap). `monogrape` keeps
    only single-varietal wines — the default, since "best Pinot Noir to drink
    now" has no clean answer for a blend.
    """

    grape: str | None = None
    region: str | None = None
    max_vintage_age: int | None = None
    monogrape: bool = True


@dataclasses.dataclass(frozen=True)
class Bundles:
    """The six trained bundles the recommender scores with, loaded once."""

    rating: ModelBundle
    q_lo: ModelBundle
    q_hi: ModelBundle
    body: ModelBundle
    acidity: ModelBundle
    harmonize: ModelBundle


@dataclasses.dataclass(frozen=True)
class DrinkNowReport:
    """Summary returned by `recommend_drink_now`."""

    table: pl.DataFrame
    path: Path
    opening_year: int
    n_candidates: int


def load_recommend_bundles() -> Bundles:
    """Load the rating, quantile, profile, and harmonize bundles from disk."""
    return Bundles(
        rating=load_bundle(RATING_BUNDLE),
        q_lo=load_bundle(RATING_QUANTILE_LO_BUNDLE),
        q_hi=load_bundle(RATING_QUANTILE_HI_BUNDLE),
        body=load_bundle(BODY_BUNDLE),
        acidity=load_bundle(ACIDITY_BUNDLE),
        harmonize=load_bundle(HARMONIZE_BUNDLE),
    )


# ---------------------------------------------------------------------------
# Candidate table
# ---------------------------------------------------------------------------


def build_candidates(notify_fn: NotifyFn | None = None) -> pl.DataFrame:
    """One row per `(wine_id, vintage_year)` across all three processed splits.

    Each split is loaded eagerly and collapsed to wine-vintage cells before the
    next is read — the memory pattern the trainers rely on, so the full-variant
    raw frame never coexists with a second copy. The three splits are disjoint
    in `(wine_id, vintage_year)`, so a plain concat is a union. Wine / winery
    names and the monogrape flag are joined from the raw wines parquet (the
    processed table carries neither).
    """
    frames: list[pl.DataFrame] = []
    for split_name in ("train", "test", "future_vintage_test"):
        notify(notify_fn, f"... gathering {split_name} candidates")
        raw = load_split(split_name)
        if not raw.is_empty():
            frames.append(aggregate_wine_vintage(raw))
        del raw
        gc.collect()
    candidates = pl.concat(frames)
    del frames
    gc.collect()

    wines = (
        pl.scan_parquet(get_settings().xwines_wines_parquet)
        .select(
            pl.col("WineID").alias("wine_id"),
            pl.col("WineName").alias("wine_name"),
            pl.col("WineryName").alias("winery_name"),
            (pl.col("Grapes").list.len() == 1).alias("is_monogrape"),
        )
        .collect()
    )
    candidates = candidates.join(wines, on="wine_id", how="left")
    # The observed cell mean is metadata, not a feature — name it so it can sit
    # in output tables next to the prediction without shadowing the model target.
    return candidates.rename({RATING_TARGET: "observed_mean_rating"})


def apply_filters(
    candidates: pl.DataFrame, filters: RecommendFilters, opening_year: int
) -> pl.DataFrame:
    """Restrict candidates by grape / region / age and drop not-yet-existing wines.

    Applied before scoring so the model only runs on wines that survive. Composes
    the year-independent filters (grape / region / monogrape) with the
    year-dependent ones (`vintage_year <= opening_year`, `max_vintage_age`), so
    standout-of-the-year can reuse the static half once and vary only the vintage
    half across years.
    """
    static = apply_static_filters(candidates, filters)
    return apply_vintage_filters(static, filters, opening_year)


def apply_static_filters(candidates: pl.DataFrame, filters: RecommendFilters) -> pl.DataFrame:
    """The year-independent filters: grape, region, and the monogrape restriction."""
    out = candidates
    if filters.monogrape:
        out = out.filter(pl.col("is_monogrape"))
    if filters.grape is not None:
        out = out.filter(
            pl.col("grape_majority").str.to_lowercase() == _slug_to_name(filters.grape)
        )
    if filters.region is not None:
        out = out.filter(pl.col("region_name").str.to_lowercase() == _slug_to_name(filters.region))
    return out


def apply_vintage_filters(
    candidates: pl.DataFrame, filters: RecommendFilters, opening_year: int
) -> pl.DataFrame:
    """The year-dependent filters: a wine can't be opened before its vintage.

    The `vintage_year <= opening_year` guard is unconditional (it also keeps swept
    ages non-negative); `max_vintage_age` additionally caps how old a wine may be
    at the opening year (the fresh-style filter).
    """
    out = candidates.filter(pl.col("vintage_year") <= opening_year)
    if filters.max_vintage_age is not None:
        out = out.filter((opening_year - pl.col("vintage_year")) <= filters.max_vintage_age)
    return out


def train_age_bounds() -> tuple[int, int]:
    """`(min, max)` of `age_at_review` on the train split — the clip range.

    Read straight off the parquet column so the recommender never has to
    materialize the raw training rows. Ages beyond this range are extrapolation;
    scoring clips to it and flags the row (`age_clipped`).
    """
    row = (
        pl.scan_parquet(split_path("train"))
        .select(pl.col(AGE_COL).min().alias("lo"), pl.col(AGE_COL).max().alias("hi"))
        .collect()
    )
    return int(row.get_column("lo")[0]), int(row.get_column("hi")[0])


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_at_opening_year(
    candidates: pl.DataFrame,
    opening_year: int,
    bundles: Bundles,
    age_bounds: tuple[int, int],
) -> pl.DataFrame:
    """Predict rating + `_lo`/`_hi` bands for every candidate at `opening_year`.

    Sets `age_at_review = opening_year − vintage_year`, clipped to the trained
    range (`age_clipped` flags a clipped row). One CatBoost pool feeds all three
    heads. Quantile crossing (`_lo > _hi`) is left as-is — the recommender
    reports the model's uncertainty, it does not paper over it.
    """
    lo, hi = age_bounds
    scored = candidates.with_columns(
        (pl.lit(opening_year) - pl.col("vintage_year")).alias("_age_raw")
    ).with_columns(
        pl.col("_age_raw").clip(lo, hi).cast(pl.Int64).alias(AGE_COL),
        ((pl.col("_age_raw") < lo) | (pl.col("_age_raw") > hi)).alias("age_clipped"),
    )
    spec = bundles.rating.feature_spec()
    pool = build_pool(scored, spec, label=None)
    return scored.with_columns(
        pl.Series("predicted_rating", bundles.rating.model.predict(pool)),
        pl.Series("predicted_rating_lo", bundles.q_lo.model.predict(pool)),
        pl.Series("predicted_rating_hi", bundles.q_hi.model.predict(pool)),
    ).drop("_age_raw")


def enrich_profile(
    candidates: pl.DataFrame, bundles: Bundles, top_pairings: int = RECOMMEND_TOP_PAIRINGS
) -> pl.DataFrame:
    """Add predicted body, acidity, and top-N food pairings (age-independent).

    These labels are wine-level constants, so they are predicted once per
    candidate set rather than once per swept opening year. The pairing list is
    the highest-probability `top_pairings` labels with the `pair_` prefix
    stripped — the same construction as `eval/sanity.py`.
    """
    body_pred = (
        np.asarray(
            bundles.body.model.predict(
                build_pool(candidates, bundles.body.feature_spec(), label=None)
            )
        )
        .ravel()
        .astype(str)
    )
    acidity_pred = (
        np.asarray(
            bundles.acidity.model.predict(
                build_pool(candidates, bundles.acidity.feature_spec(), label=None)
            )
        )
        .ravel()
        .astype(str)
    )

    labels = list(bundles.harmonize.meta["targets"])
    proba = np.asarray(
        bundles.harmonize.model.predict_proba(
            build_pool(candidates, bundles.harmonize.feature_spec(), label=None)
        )
    )
    k = min(top_pairings, len(labels))
    stripped = [_strip_pair(label) for label in labels]
    # Stable sort so ties resolve deterministically across runs.
    top_idx = np.argsort(-proba, axis=1, kind="stable")[:, :k]
    top_lists = [[stripped[j] for j in row] for row in top_idx]

    return candidates.with_columns(
        pl.Series("predicted_body", body_pred),
        pl.Series("predicted_acidity", acidity_pred),
        pl.Series("top_pairings", top_lists, dtype=pl.List(pl.Utf8)),
    )


# ---------------------------------------------------------------------------
# Drink-now ranking
# ---------------------------------------------------------------------------


def recommend_drink_now(
    *,
    opening_year: int = DEFAULT_OPENING_YEAR,
    filters: RecommendFilters | None = None,
    top: int = RECOMMEND_TOP_N,
    distinct_wines: bool = False,
    out_path: Path | None = None,
    candidates: pl.DataFrame | None = None,
    bundles: Bundles | None = None,
    notify_fn: NotifyFn | None = None,
) -> DrinkNowReport:
    """Rank wines by predicted rating at a single opening year.

    Scores every filtered candidate at `age_at_review = opening_year −
    vintage_year`, sorts by predicted rating, and writes the top-`top` rows to
    `out_path` (default: the configured drink-now parquet). The returned table
    is the same top-`top` slice, for the CLI to print.

    `distinct_wines=True` collapses the ranking to one row per wine (its
    best-scoring vintage) before taking the top-`top`. Producer identity
    dominates the model, so the raw ranking is often ten vintages of a single
    estate; the collapsed view shows ten different wines and is what the
    published tables use.

    Pass `candidates` (the raw, unfiltered `build_candidates()` frame) and
    `bundles` to reuse them across many calls — the generator scores dozens of
    grape slices without rebuilding the candidate table each time. This call
    still applies its own filters.
    """
    filters = filters or RecommendFilters()
    settings = get_settings()
    out_path = out_path or settings.recommendations_drink_now_parquet

    bundles = bundles or load_recommend_bundles()
    raw = candidates if candidates is not None else build_candidates(notify_fn)
    candidates = apply_filters(raw, filters, opening_year)
    if candidates.is_empty():
        raise ValueError(
            f"No wines match the filters at opening year {opening_year}. "
            "Loosen --grape / --region / --max-vintage-age, or drop --monogrape."
        )

    notify(notify_fn, f"... scoring {candidates.height:,} wines at {opening_year}")
    age_bounds = train_age_bounds()
    scored = score_at_opening_year(candidates, opening_year, bundles, age_bounds)
    scored = enrich_profile(scored, bundles)

    # Deterministic tiebreak (wine, vintage) so the ranking is reproducible and
    # matches standout-of-the-year's per-year ordering row-for-row.
    ranked = scored.sort(
        ["predicted_rating", "wine_id", "vintage_year"], descending=[True, False, False]
    )
    if distinct_wines:
        ranked = collapse_distinct_wines(ranked)
    table = ranked.head(top).select(DRINK_NOW_OUT_COLS)
    write_ranking_parquet(table, out_path)
    notify(notify_fn, f"... wrote {table.height} rows to {out_path}")
    return DrinkNowReport(
        table=table, path=out_path, opening_year=opening_year, n_candidates=candidates.height
    )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def collapse_distinct_wines(ranked: pl.DataFrame) -> pl.DataFrame:
    """Keep one row per `wine_id` — the first, i.e. best, in an already-sorted frame.

    Producer identity is the single strongest signal in the rating model, so a
    grape's raw ranking is frequently the same estate's wine across a dozen
    vintages. Collapsing to the best vintage per wine turns a ten-row table of
    one wine into ten distinct wines. `maintain_order=True` preserves the caller's
    sort, so "best" is whatever the caller ranked first.
    """
    return ranked.unique(subset="wine_id", keep="first", maintain_order=True)


def write_ranking_parquet(df: pl.DataFrame, path: Path) -> None:
    """Write a ranking parquet atomically (tmp → rename), like `artifacts._atomic_write`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.write_parquet(tmp)
    tmp.replace(path)


def _slug_to_name(slug: str) -> str:
    """`"pinot-noir"` / `"pinot_noir"` → `"pinot noir"` for case-folded matching."""
    return slug.strip().lower().replace("-", " ").replace("_", " ")


def _strip_pair(column: str) -> str:
    """`pair_beef` → `beef` for display."""
    return column.removeprefix(HARMONIZE_TARGET_PREFIX)
