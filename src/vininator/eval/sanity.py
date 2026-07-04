"""Qualitative sanity check (Phase 5): score wines the user knows personally.

Looks up wines by name / winery substring, pulls their already-assembled
feature rows from the processed splits (never re-engineers features — same
rule as the Phase 6 recommender), and predicts rating, body, acidity, and
food pairings next to the observed labels. The point is the eyeball test:
disagreements between the model and someone who has actually drunk the wine
are written up in RESULTS.md §8, and they're more interesting than the
agreements.

Each result row says which split the wine's rows live in — a `train` wine's
prediction is a fitted value, not a forecast, and the writeup must read it
that way.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import polars as pl

from vininator.config import (
    ACIDITY_BUNDLE,
    BODY_BUNDLE,
    HARMONIZE_BUNDLE,
    HARMONIZE_TARGET_PREFIX,
    RATING_BUNDLE,
    RATING_TARGET,
    get_settings,
)
from vininator.models.artifacts import ModelBundle, load_bundle
from vininator.models.dataset import (
    NotifyFn,
    aggregate_rating_cells,
    aggregate_wine_vintage,
    build_pool,
    notify,
)

_MAX_MATCHES_PER_QUERY = 3


@dataclasses.dataclass(frozen=True)
class SanityRow:
    """One wine-vintage scored next to its observed labels."""

    query: str
    wine_id: int
    wine_name: str
    winery_name: str
    region_name: str
    vintage_year: int
    split: str
    age_at_review: int
    n_ratings: int
    observed_mean_rating: float
    predicted_rating: float
    actual_body: str | None
    predicted_body: str
    actual_acidity: str | None
    predicted_acidity: str
    predicted_pairings: list[str]
    actual_pairings: list[str]


@dataclasses.dataclass(frozen=True)
class SanityReport:
    """Summary returned by `sanity_check`."""

    rows: list[SanityRow]
    unmatched: list[str]


def sanity_check(
    queries: list[str],
    *,
    top_pairings: int = 5,
    notify_fn: NotifyFn | None = None,
) -> SanityReport:
    """Predict rating + profile + pairings for wines matched by name.

    Args:
        queries: Case-insensitive substrings matched against
            `"{WineryName} {WineName}"` in the raw wines table; each query
            contributes at most `_MAX_MATCHES_PER_QUERY` wines.
        top_pairings: How many highest-probability pairings to report.
        notify_fn: Milestone callback (CLI passes `typer.echo`).

    Returns:
        A `SanityReport`; queries that matched nothing land in `unmatched`.
    """
    settings = get_settings()
    wines = (
        pl.scan_parquet(settings.xwines_wines_parquet)
        .select("WineID", "WineName", "WineryName", "RegionName")
        .with_columns(
            pl.concat_str([pl.col("WineryName"), pl.lit(" "), pl.col("WineName")])
            .str.to_lowercase()
            .alias("_haystack")
        )
        .collect()
    )

    matches: list[tuple[str, int, str, str, str]] = []
    unmatched: list[str] = []
    for query in queries:
        hit = wines.filter(pl.col("_haystack").str.contains(query.lower(), literal=True))
        if hit.is_empty():
            unmatched.append(query)
            continue
        if hit.height > _MAX_MATCHES_PER_QUERY:
            notify(
                notify_fn,
                f"... {query!r} matched {hit.height} wines; keeping the first "
                f"{_MAX_MATCHES_PER_QUERY}",
            )
            hit = hit.head(_MAX_MATCHES_PER_QUERY)
        matches.extend(
            (query, row["WineID"], row["WineName"], row["WineryName"], row["RegionName"])
            for row in hit.iter_rows(named=True)
        )
    if not matches:
        return SanityReport(rows=[], unmatched=unmatched)

    wine_ids = [m[1] for m in matches]
    notify(notify_fn, f"... gathering processed rows for {len(wine_ids)} wines")
    rows = _gather_rows(wine_ids)

    rating = load_bundle(RATING_BUNDLE)
    body = load_bundle(BODY_BUNDLE)
    acidity = load_bundle(ACIDITY_BUNDLE)
    harmonize = load_bundle(HARMONIZE_BUNDLE)

    out: list[SanityRow] = []
    for query, wine_id, wine_name, winery_name, region_name in matches:
        wine_rows = rows.filter(pl.col("wine_id") == wine_id)
        if wine_rows.is_empty():
            unmatched.append(f"{query} (wine_id={wine_id} has no processed rows)")
            continue
        out.append(
            _score_wine(
                wine_rows,
                query=query,
                wine_id=wine_id,
                wine_name=wine_name,
                winery_name=winery_name,
                region_name=region_name,
                rating=rating,
                body=body,
                acidity=acidity,
                harmonize=harmonize,
                top_pairings=top_pairings,
            )
        )
    return SanityReport(rows=out, unmatched=unmatched)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _gather_rows(wine_ids: list[int]) -> pl.DataFrame:
    """All processed rows for the given wines, across the three splits."""
    settings = get_settings()
    frames = [
        pl.scan_parquet(path).filter(pl.col("wine_id").is_in(wine_ids)).collect()
        for path in (
            settings.processed_train_parquet,
            settings.processed_test_parquet,
            settings.processed_future_vintage_test_parquet,
        )
    ]
    return pl.concat(frames)


def _score_wine(
    wine_rows: pl.DataFrame,
    *,
    query: str,
    wine_id: int,
    wine_name: str,
    winery_name: str,
    region_name: str,
    rating: ModelBundle,
    body: ModelBundle,
    acidity: ModelBundle,
    harmonize: ModelBundle,
    top_pairings: int,
) -> SanityRow:
    """Score the wine's most-rated vintage at its most-rated review age."""
    top_vintage = (
        wine_rows.group_by("vintage_year")
        .len()
        .sort("len", descending=True)
        .get_column("vintage_year")[0]
    )
    vintage_rows = wine_rows.filter(pl.col("vintage_year") == top_vintage)

    # Rating: predict the most-rated (wine, vintage, age) cell.
    cells = aggregate_rating_cells(vintage_rows).sort("n_ratings_cell", descending=True)
    cell = cells.head(1)
    predicted_rating = float(
        rating.model.predict(build_pool(cell, rating.feature_spec(), label=None))[0]
    )

    # Profile + pairings: one wine-vintage row (labels are wine-level).
    wv = aggregate_wine_vintage(vintage_rows).head(1)
    predicted_body = str(body.model.predict(build_pool(wv, body.feature_spec(), label=None))[0][0])
    predicted_acidity = str(
        acidity.model.predict(build_pool(wv, acidity.feature_spec(), label=None))[0][0]
    )

    labels: list[str] = list(harmonize.meta["targets"])
    proba = np.asarray(
        harmonize.model.predict_proba(build_pool(wv, harmonize.feature_spec(), label=None))
    )[0]
    ranked = sorted(zip(labels, proba, strict=True), key=lambda kv: kv[1], reverse=True)[
        :top_pairings
    ]
    predicted_pairings = [_strip_pair(label) for label, _ in ranked]
    actual_pairings = [
        _strip_pair(c) for c in labels if c in wv.columns and wv.get_column(c)[0] == 1
    ]

    return SanityRow(
        query=query,
        wine_id=wine_id,
        wine_name=wine_name,
        winery_name=winery_name,
        region_name=region_name,
        vintage_year=int(top_vintage),
        split=str(cell.get_column("split")[0]),
        age_at_review=int(cell.get_column("age_at_review")[0]),
        n_ratings=int(cell.get_column("n_ratings_cell")[0]),
        observed_mean_rating=float(cell.get_column(RATING_TARGET)[0]),
        predicted_rating=predicted_rating,
        actual_body=wv.get_column("body_label")[0],
        predicted_body=predicted_body,
        actual_acidity=wv.get_column("acidity_label")[0],
        predicted_acidity=predicted_acidity,
        predicted_pairings=predicted_pairings,
        actual_pairings=actual_pairings,
    )


def _strip_pair(column: str) -> str:
    """`pair_beef` → `beef` for display."""
    return column.removeprefix(HARMONIZE_TARGET_PREFIX)
