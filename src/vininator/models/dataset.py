"""The feature contract shared by every Phase 4 model.

One module decides how the 127-column processed table (features/build.py) is
sliced into CatBoost inputs, so the rating, profile, and harmonize trainers all
agree on what is a feature, what is a target, and what is categorical. Changing
the contract here changes it everywhere at once.

Separation of concerns:
- `load_split` / `load_train_config`: I/O — read parquets and yaml configs.
- everything else: pure — schema arithmetic, frame preparation, cell
  aggregation, the grouped-by-wine validation split. No global state, no
  network.

Trainers never fit on raw rating rows: they collapse to feature cells first
(`aggregate_rating_cells` for the rating model, `aggregate_wine_vintage` for
the wine-level targets). See those functions for why this is loss-exact for
RMSE and 7–21× cheaper.

CatBoost specifics handled here, once:
- Categoricals must be non-null strings → fill nulls with a sentinel, cast
  high-cardinality int keys (winery_id) to Utf8.
- Booleans become 0/1 floats so a missing region's flags stay NaN (CatBoost
  reads NaN in *numeric* columns natively but rejects it in categoricals).
- The frame is handed to CatBoost as pandas (the only place pandas appears);
  polars owns everything upstream.
"""

from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import polars as pl
import yaml
from catboost import Pool

from vininator.config import (
    AGE_COL,
    BOOL_FEATURE_COLS,
    CATEGORICAL_FEATURE_COLS,
    CATEGORICAL_NULL_SENTINEL,
    CELL_N_RATINGS_COL,
    GROUP_KEY_COL,
    GROUP_VAL_FRAC,
    HARMONIZE_TARGET_PREFIX,
    MODEL_METADATA_COLS,
    MODEL_SEED,
    RATING_CELL_KEYS,
    RATING_TARGET,
    SAMPLE_WEIGHT_COL,
    WINE_VINTAGE_KEYS,
    get_settings,
)

SplitName = Literal["train", "test", "future_vintage_test"]
NotifyFn = Callable[[str], None]


def notify(notify_fn: NotifyFn | None, message: str) -> None:
    """Invoke a milestone callback if one was supplied (no-op otherwise)."""
    if notify_fn is not None:
        notify_fn(message)


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    """Parsed experiment yaml: CatBoost params plus the data subset to train on."""

    name: str
    catboost_params: dict[str, Any]
    sample_frac: float | None
    max_rows: int | None
    seed: int


@dataclasses.dataclass(frozen=True)
class FeatureSpec:
    """The feature columns and the categorical subset for one model's inputs."""

    feature_cols: list[str]
    cat_cols: list[str]


# ---------------------------------------------------------------------------
# Config + split I/O
# ---------------------------------------------------------------------------


def load_train_config(path: Path) -> TrainConfig:
    """Read an experiment yaml into a `TrainConfig`.

    The yaml shape is::

        catboost_params: {iterations: ..., depth: ..., ...}
        data: {sample_frac: 0.01, max_rows: null, seed: 42}

    Missing keys fall back to sensible defaults so a minimal config still runs.
    """
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    data = raw.get("data", {}) or {}
    return TrainConfig(
        name=raw.get("name", path.stem),
        catboost_params=dict(raw.get("catboost_params", {}) or {}),
        sample_frac=data.get("sample_frac"),
        max_rows=data.get("max_rows"),
        seed=int(data.get("seed", MODEL_SEED)),
    )


def resolve_catboost_params(cfg: TrainConfig, *, loss_function: str) -> dict[str, Any]:
    """Merge config CatBoost params with the project defaults for one loss.

    The seed and `allow_writing_files=False` (suppress CatBoost's `catboost_info/`
    scratch dir) are always set; the caller-chosen `loss_function` always wins so
    the quantile heads can reuse the same config block as the RMSE model.
    """
    params: dict[str, Any] = {
        "random_seed": cfg.seed,
        "allow_writing_files": False,
    }
    params.update(cfg.catboost_params)
    params["loss_function"] = loss_function
    return params


def split_path(split: SplitName) -> Path:
    """Resolve a split name to its processed parquet path."""
    settings = get_settings()
    return {
        "train": settings.processed_train_parquet,
        "test": settings.processed_test_parquet,
        "future_vintage_test": settings.processed_future_vintage_test_parquet,
    }[split]


def load_split(
    split: SplitName,
    *,
    sample_frac: float | None = None,
    max_rows: int | None = None,
) -> pl.DataFrame:
    """Load one processed split, optionally subsampled for fast smoke runs.

    Subsampling is by `wine_id` hash, not by row: a wine is either wholly in or
    wholly out. This keeps the subsample consistent with the wine-level grouping
    the models rely on and lets polars push the filter into the parquet scan.

    Args:
        split: Which processed parquet to read.
        sample_frac: Keep wines whose hash falls in the first `frac` of the
            hash space (deterministic). `None` keeps everything.
        max_rows: Hard cap applied after sampling (a final `head`).

    Returns:
        The split as an eager polars DataFrame.
    """
    lf = pl.scan_parquet(split_path(split))
    if sample_frac is not None:
        if not 0.0 < sample_frac <= 1.0:
            raise ValueError(f"sample_frac must be in (0, 1], got {sample_frac}")
        cutoff = int(sample_frac * 10_000)
        lf = lf.filter((pl.col(GROUP_KEY_COL).hash(seed=MODEL_SEED) % 10_000) < cutoff)
    if max_rows is not None:
        # head() pushed into the lazy plan lets polars skip unread row-groups —
        # reads only as many parquet pages as needed, not the full file.
        lf = lf.head(max_rows)
    return lf.collect()


# ---------------------------------------------------------------------------
# Feature contract (pure)
# ---------------------------------------------------------------------------


def harmonize_target_cols(schema_names: list[str]) -> list[str]:
    """The `pair_*` multi-hot columns — the harmonize model's targets."""
    return [c for c in schema_names if c.startswith(HARMONIZE_TARGET_PREFIX)]


def feature_spec(schema_names: list[str], *, targets: list[str]) -> FeatureSpec:
    """Derive the feature set for a model from the full column list.

    Features = all columns − metadata (`MODEL_METADATA_COLS`, which already
    excludes `rating`) − this model's own target columns. The categorical
    subset is whatever survives from `CATEGORICAL_FEATURE_COLS`, so a model
    whose target is `body_label` simply drops it from both lists.
    """
    excluded = set(MODEL_METADATA_COLS) | set(targets)
    feature_cols = [c for c in schema_names if c not in excluded]
    cat_cols = [c for c in feature_cols if c in CATEGORICAL_FEATURE_COLS]
    return FeatureSpec(feature_cols=feature_cols, cat_cols=cat_cols)


def prepare_features(df: pl.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    """Project to feature columns and make them CatBoost-safe, as pandas.

    Categoricals are filled and stringified; booleans become 0/1 floats; every
    other column passes through (CatBoost tolerates NaN in numerics). This is
    the single polars→pandas boundary in the modeling code.
    """
    casts: list[pl.Expr] = []
    for col in spec.feature_cols:
        if col in spec.cat_cols:
            casts.append(pl.col(col).cast(pl.Utf8).fill_null(CATEGORICAL_NULL_SENTINEL).alias(col))
        elif col in BOOL_FEATURE_COLS:
            casts.append(pl.col(col).cast(pl.Int8).cast(pl.Float64).alias(col))
    prepared = df.with_columns(casts).select(spec.feature_cols)
    out = prepared.to_pandas()
    # Belt-and-suspenders: converting a large mixed-type frame can inject float
    # NaN into an otherwise null-free String column on the pandas side, which
    # CatBoost rejects ("NaN values should be converted to string"). Re-fill the
    # categoricals here so they are guaranteed null-free strings at the boundary.
    for col in spec.cat_cols:
        out[col] = out[col].fillna(CATEGORICAL_NULL_SENTINEL)
    return out


def build_pool(
    df: pl.DataFrame,
    spec: FeatureSpec,
    *,
    label: pd.Series | pd.DataFrame | None,
    weight: pd.Series | None = None,
) -> Pool:
    """Assemble a CatBoost `Pool` from a prepared feature frame.

    `label` is a 1-D series for regression / multiclass, a 2-D frame for the
    multilabel harmonize model, or `None` for an inference-only pool.
    """
    features = prepare_features(df, spec)
    return Pool(
        data=features,
        label=label,
        weight=weight,
        cat_features=spec.cat_cols,
    )


def sample_weights(df: pl.DataFrame) -> pd.Series:
    """The per-row training weight column (`log(1 + n_ratings)`) as pandas."""
    return df.get_column(SAMPLE_WEIGHT_COL).to_pandas()


# ---------------------------------------------------------------------------
# Cell aggregation (pure)
# ---------------------------------------------------------------------------
#
# Every feature in the processed table is constant within its aggregation
# cell: wine-level attributes are keyed by wine_id, terroir by (region,
# vintage) — and region is a function of wine_id — and the age itself is a
# key. `tests/models/test_dataset.py::test_rating_cells_have_constant_features`
# guards this invariant; any new feature that varies inside a cell must either
# extend the cell keys or become metadata.

_NON_AGGREGATED_COLS = ("rating_id", "rating_date")


def _aggregate(df: pl.DataFrame, keys: list[str], *, drop: tuple[str, ...] = ()) -> pl.DataFrame:
    """Collapse rows sharing `keys`: mean rating, summed weight, count, first-of-rest."""
    skip = {*keys, RATING_TARGET, SAMPLE_WEIGHT_COL, *_NON_AGGREGATED_COLS, *drop}
    passthrough = [c for c in df.columns if c not in skip]
    return (
        df.group_by(keys)
        .agg(
            pl.col(RATING_TARGET).mean(),
            pl.col(SAMPLE_WEIGHT_COL).sum(),
            pl.len().alias(CELL_N_RATINGS_COL),
            *[pl.col(c).first() for c in passthrough],
        )
        # CatBoost results depend on row order (ordered statistics, bootstrap);
        # sort so a fresh run reproduces the same model bit-for-bit.
        .sort(keys)
    )


def aggregate_rating_cells(df: pl.DataFrame) -> pl.DataFrame:
    """One row per (wine, vintage, age) feature cell for the rating model.

    For weighted squared loss this is *gradient-identical* to training on the
    raw rating rows: with constant per-row weight w inside a cell of n rows,
    Σ w·(yᵢ − p)² = (n·w)·(ȳ − p)² + constant, so the fitted trees, the
    early-stopping argmin, and the final model are unchanged while the full
    variant shrinks ~7× (15.5M rows → 2.25M cells). The quantile heads change
    meaning *deliberately*: trained on cells they band the wine-vintage mean
    rating (the recommender's actual question) instead of the spread of
    individual user opinions, which was so wide it made the overperformer
    filter useless.

    `rating` becomes the cell mean (weights are constant within a cell, so the
    plain mean is the weighted mean), `sample_weight` the cell sum, and
    `n_ratings_cell` carries the collapse count for eval weighting (metadata,
    never a feature).
    """
    return _aggregate(df, list(RATING_CELL_KEYS))


def aggregate_wine_vintage(df: pl.DataFrame) -> pl.DataFrame:
    """One row per (wine, vintage) for the wine-level targets (body, acidity, pair_*).

    Those labels are constant per wine, so per-rating rows are pure duplication
    (~21× on the full variant: 15.5M rows → 746k wine-vintages) and per-rating
    eval double-counts popular wines. `age_at_review` is dropped entirely — a wine-level label cannot
    depend on when reviewers happened to rate, and keeping it would invite the
    model to fit noise.
    """
    return _aggregate(df, list(WINE_VINTAGE_KEYS), drop=(AGE_COL,))


# ---------------------------------------------------------------------------
# Grouped validation split (pure)
# ---------------------------------------------------------------------------


def grouped_val_split(
    df: pl.DataFrame,
    *,
    frac: float = GROUP_VAL_FRAC,
    seed: int = MODEL_SEED,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Carve an early-stopping validation fold out of train, split by wine.

    Mirrors `features.build._assign_test_wine_ids`: sort the unique wine_ids,
    seeded-shuffle, take the first `frac` as validation. No `wine_id` straddles
    the fold boundary, so early stopping never sees a row whose wine is in the
    fit set.

    Returns:
        `(fit_df, val_df)`. If there is only one wine, `val_df` is empty and the
        caller should train without an eval set.
    """
    ids = df.get_column(GROUP_KEY_COL).unique().sort().shuffle(seed=seed)
    n_val = int(len(ids) * frac)
    if n_val == 0 or n_val == len(ids):
        return df, df.clear()
    val_ids = ids[:n_val].implode()
    val = df.filter(pl.col(GROUP_KEY_COL).is_in(val_ids))
    fit = df.filter(~pl.col(GROUP_KEY_COL).is_in(val_ids))
    return fit, val


# ---------------------------------------------------------------------------
# Tracking helpers (pure)
# ---------------------------------------------------------------------------


def dataset_hash(df: pl.DataFrame) -> str:
    """A short, cheap fingerprint of a dataset's shape and schema.

    Hashes row count plus the ordered `(name: dtype)` schema — enough to detect
    "this run trained on a different table" in MLflow without paying to hash
    15M rows of content.
    """
    h = hashlib.sha256()
    h.update(str(df.height).encode())
    for name, dtype in df.schema.items():
        h.update(f"{name}:{dtype}".encode())
    return h.hexdigest()[:16]
