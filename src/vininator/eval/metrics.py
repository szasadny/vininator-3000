"""Metrics and leakage-safe rating baselines.

Two kinds of thing live here, both pure:

- **Metric functions** (`rmse`, `mae`, `macro_f1`, `per_label_f1`, `hamming`):
  thin, well-tested wrappers operating on numpy / sklearn. Rating metrics are
  unweighted so they line up 1:1 with the PROJECT.md baseline numbers (global
  mean RMSE = std(Rating)); the sample weight only ever shapes the *loss*, not
  the report.

- **Baselines** (`fit_rating_baselines` / `apply_rating_baselines` and the
  legacy `rating_baselines` wrapper): each fits a group-mean lookup on the
  **train** frame and predicts on an eval frame, falling back to the train
  global mean for groups unseen in train. They are leakage-safe under both the
  wine split and the future-vintage split — the eval frame's ratings never enter
  the lookup. The per-(region, vintage) and per-(grape, region) baselines are
  the numbers the terroir model actually has to beat.

  Use the split API (`fit_rating_baselines` then `apply_rating_baselines`) in
  training code that needs to free the large train frame before evaluation; the
  resulting `RatingBaselineFit` is tiny (a few MB at most).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import numpy as np
import polars as pl
from numpy.typing import ArrayLike
from sklearn.metrics import f1_score, hamming_loss

_RATING_COL = "rating"


@dataclasses.dataclass(frozen=True)
class BaselineResult:
    """RMSE/MAE for one baseline on one eval split."""

    name: str
    rmse: float
    mae: float


@dataclasses.dataclass(frozen=True)
class RatingBaselineFit:
    """Pre-computed group-mean lookups fit on the train frame.

    Holding this instead of the full train frame costs KBs, not GBs — safe to
    keep alive for the whole training run while the source frame is freed.
    """

    global_mean: float
    winery_lookup: pl.DataFrame  # (winery_id, _pred)
    region_vintage_lookup: pl.DataFrame  # (region_name, vintage_year, _pred)
    grape_region_lookup: pl.DataFrame  # (grape_majority, region_name, _pred)


# ---------------------------------------------------------------------------
# Regression metrics
# ---------------------------------------------------------------------------


def rmse(y_true: ArrayLike, y_pred: ArrayLike, weight: ArrayLike | None = None) -> float:
    """Root mean squared error, optionally sample-weighted."""
    err2 = (np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)) ** 2
    if weight is None:
        return float(np.sqrt(err2.mean()))
    return float(np.sqrt(np.average(err2, weights=np.asarray(weight, dtype=float))))


def mae(y_true: ArrayLike, y_pred: ArrayLike, weight: ArrayLike | None = None) -> float:
    """Mean absolute error, optionally sample-weighted."""
    err = np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))
    if weight is None:
        return float(err.mean())
    return float(np.average(err, weights=np.asarray(weight, dtype=float)))


def within_group_std(df: pl.DataFrame, keys: Sequence[str], col: str = _RATING_COL) -> float:
    """Pooled within-group standard deviation of `col`, grouped by `keys`.

    This is the irreducible per-rating RMSE floor for any model whose features
    are constant within the groups: even a perfect model predicts one value per
    cell and eats the within-cell spread as error. On the full X-Wines train
    split the floor is ~0.64 against a global std of ~0.74 — which is why
    per-rating RMSE is reported next to this number instead of being read as an
    absolute score. Groups with a single row carry no spread information and
    are skipped.
    """
    cells = (
        df.group_by(list(keys))
        .agg(pl.col(col).var().alias("_var"), pl.len().alias("_n"))
        .filter(pl.col("_n") > 1)
    )
    if cells.is_empty():
        return 0.0
    dof = (cells.get_column("_n") - 1).sum()
    pooled = ((cells.get_column("_var") * (cells.get_column("_n") - 1)).sum()) / dof
    return float(np.sqrt(pooled))


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------


def macro_f1(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Macro-averaged F1 — the headline metric for the skewed profile labels."""
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def per_label_f1(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> dict[str, float]:
    """Binary F1 per column of a multilabel prediction matrix."""
    return {
        label: float(f1_score(y_true[:, i], y_pred[:, i], zero_division=0))
        for i, label in enumerate(labels)
    }


def hamming(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Hamming loss for a multilabel prediction matrix (lower is better)."""
    return float(hamming_loss(y_true, y_pred))


# ---------------------------------------------------------------------------
# Rating baselines (leakage-safe: fit on train, predict on eval)
# ---------------------------------------------------------------------------


def fit_rating_baselines(train: pl.DataFrame) -> RatingBaselineFit:
    """Pre-compute group-mean lookups from the train frame.

    Call this before freeing the large train frame; pass the returned
    `RatingBaselineFit` to `apply_rating_baselines` at eval time.
    """
    global_mean = float(train.get_column(_RATING_COL).mean())
    return RatingBaselineFit(
        global_mean=global_mean,
        winery_lookup=_fit_grouped_mean(train, ["winery_id"]),
        region_vintage_lookup=_fit_grouped_mean(train, ["region_name", "vintage_year"]),
        grape_region_lookup=_fit_grouped_mean(train, ["grape_majority", "region_name"]),
    )


def apply_rating_baselines(
    fit: RatingBaselineFit, eval_df: pl.DataFrame, *, weight_col: str | None = None
) -> list[BaselineResult]:
    """Evaluate all baselines on one eval split using pre-fit lookups.

    Report order: global_mean, winery_mean, region_vintage_mean, grape_region_mean.
    Pass `weight_col` when `eval_df` holds aggregated cells rather than raw
    rating rows, so the baseline numbers stay comparable to the model's
    cell-level metrics (weighted by ratings per cell).
    """
    y = eval_df.get_column(_RATING_COL).to_numpy()
    w = eval_df.get_column(weight_col).to_numpy() if weight_col is not None else None
    global_pred = np.full(y.shape, fit.global_mean, dtype=float)
    return [
        BaselineResult("global_mean", rmse(y, global_pred, w), mae(y, global_pred, w)),
        _apply_grouped_mean(
            fit.winery_lookup, ["winery_id"], fit.global_mean, eval_df, "winery_mean", weight_col
        ),
        _apply_grouped_mean(
            fit.region_vintage_lookup,
            ["region_name", "vintage_year"],
            fit.global_mean,
            eval_df,
            "region_vintage_mean",
            weight_col,
        ),
        _apply_grouped_mean(
            fit.grape_region_lookup,
            ["grape_majority", "region_name"],
            fit.global_mean,
            eval_df,
            "grape_region_mean",
            weight_col,
        ),
    ]


def _fit_grouped_mean(train: pl.DataFrame, keys: list[str]) -> pl.DataFrame:
    """Group-mean lookup table from train, keyed on `keys`."""
    return train.group_by(keys).agg(pl.col(_RATING_COL).mean().alias("_pred"))


def _apply_grouped_mean(
    lookup: pl.DataFrame,
    keys: list[str],
    global_mean: float,
    eval_df: pl.DataFrame,
    name: str,
    weight_col: str | None = None,
) -> BaselineResult:
    """Join eval_df to a pre-fit lookup; fall back to global_mean for unseen groups."""
    cols = [*keys, _RATING_COL] + ([weight_col] if weight_col is not None else [])
    joined = (
        eval_df.select(cols)
        .join(lookup, on=keys, how="left")
        .with_columns(pl.col("_pred").fill_null(global_mean))
    )
    y = joined.get_column(_RATING_COL).to_numpy()
    pred = joined.get_column("_pred").to_numpy()
    w = joined.get_column(weight_col).to_numpy() if weight_col is not None else None
    return BaselineResult(name, rmse(y, pred, w), mae(y, pred, w))


# ---------------------------------------------------------------------------
# Legacy single-call API (kept for existing call sites and tests)
# ---------------------------------------------------------------------------


def baseline_global_mean(train: pl.DataFrame, eval_df: pl.DataFrame) -> BaselineResult:
    """Predict the train global mean rating for every eval row."""
    global_mean = float(train.get_column(_RATING_COL).mean())
    y = eval_df.get_column(_RATING_COL).to_numpy()
    pred = np.full(y.shape, global_mean, dtype=float)
    return BaselineResult("global_mean", rmse(y, pred), mae(y, pred))


def _grouped_mean_baseline(
    train: pl.DataFrame,
    eval_df: pl.DataFrame,
    keys: list[str],
    name: str,
) -> BaselineResult:
    """Per-group train mean, with the train global mean as the unseen-group fallback."""
    global_mean = float(train.get_column(_RATING_COL).mean())
    lookup = _fit_grouped_mean(train, keys)
    return _apply_grouped_mean(lookup, keys, global_mean, eval_df, name)


def baseline_region_vintage_mean(train: pl.DataFrame, eval_df: pl.DataFrame) -> BaselineResult:
    """Per-(region, vintage) train mean — the production number to beat."""
    return _grouped_mean_baseline(
        train, eval_df, ["region_name", "vintage_year"], "region_vintage_mean"
    )


def baseline_grape_region_mean(train: pl.DataFrame, eval_df: pl.DataFrame) -> BaselineResult:
    """Per-(majority grape, region) train mean."""
    return _grouped_mean_baseline(
        train, eval_df, ["grape_majority", "region_name"], "grape_region_mean"
    )


def baseline_winery_mean(train: pl.DataFrame, eval_df: pl.DataFrame) -> BaselineResult:
    """Per-winery train mean — the leakage-safe stand-in for per-WineID."""
    return _grouped_mean_baseline(train, eval_df, ["winery_id"], "winery_mean")


def rating_baselines(train: pl.DataFrame, eval_df: pl.DataFrame) -> list[BaselineResult]:
    """All rating baselines for one eval split, in report order."""
    return apply_rating_baselines(fit_rating_baselines(train), eval_df)
