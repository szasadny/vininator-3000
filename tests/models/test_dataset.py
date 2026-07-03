"""Tests for the shared feature contract in models/dataset.py.

The leakage-critical guarantees live here: `rating` is never a feature, each
model's target is excluded from its own inputs, categoricals are never handed
to CatBoost with nulls, and the validation fold is wine-disjoint from the fit
fold (the same boundary the train/test split enforces).
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from vininator.config import (
    CELL_N_RATINGS_COL,
    MODEL_METADATA_COLS,
    RATING_CELL_KEYS,
    RATING_TARGET,
)
from vininator.models.dataset import (
    FeatureSpec,
    aggregate_rating_cells,
    aggregate_wine_vintage,
    feature_spec,
    grouped_val_split,
    harmonize_target_cols,
    load_split,
    prepare_features,
)

_COLS = [
    "rating_id",
    "wine_id",
    "rating",
    "rating_date",
    "split",
    "sample_weight",
    "vintage_year",
    "age_at_review",
    "abv",
    "wine_type",
    "country",
    "region_name",
    "winery_id",
    "grape_majority",
    "grape_merlot",
    "grape_other",
    "pair_beef",
    "pair_pork",
    "gdd_10c",
    "is_partial",
    "drainage_class",
    "calcareous",
    "producer_mean_rating",
    "body_label",
    "acidity_label",
]


def test_rating_never_a_feature() -> None:
    """`rating` must be absent from every model's feature set."""
    for targets in ([RATING_TARGET], ["body_label"], ["acidity_label"], ["pair_beef"]):
        spec = feature_spec(_COLS, targets=targets)
        assert "rating" not in spec.feature_cols
        assert "sample_weight" not in spec.feature_cols
        assert "wine_id" not in spec.feature_cols


def test_target_excluded_but_sibling_label_kept() -> None:
    """Predicting body drops body_label but keeps acidity_label as a feature."""
    spec = feature_spec(_COLS, targets=["body_label"])
    assert "body_label" not in spec.feature_cols
    assert "acidity_label" in spec.feature_cols
    assert "acidity_label" in spec.cat_cols


def test_harmonize_targets_all_excluded() -> None:
    """Every pair_* column is a target for harmonize, none a feature."""
    targets = harmonize_target_cols(_COLS)
    assert set(targets) == {"pair_beef", "pair_pork"}
    spec = feature_spec(_COLS, targets=targets)
    assert not any(c.startswith("pair_") for c in spec.feature_cols)
    assert "body_label" in spec.feature_cols  # other blocks survive


def test_prepare_features_fills_categoricals_and_casts_bools() -> None:
    """No nulls reach CatBoost categoricals; booleans become 0/1 floats."""
    df = pl.DataFrame(
        {
            "winery_id": [1, None],
            "drainage_class": ["loamy", None],
            "calcareous": [True, None],
            "is_partial": [False, True],
            "gdd_10c": [1400.0, None],
        }
    )
    spec = FeatureSpec(feature_cols=list(df.columns), cat_cols=["winery_id", "drainage_class"])
    out = prepare_features(df, spec)

    assert out["drainage_class"].isna().sum() == 0
    assert (out["drainage_class"] == "unknown").any()
    assert (out["winery_id"] == "unknown").any()  # int key stringified + filled
    assert set(out["calcareous"].dropna().unique()) <= {0.0, 1.0}
    assert out["gdd_10c"].isna().sum() == 1  # numeric NaN preserved for CatBoost


def _raw_rating_rows() -> pl.DataFrame:
    """Six rating rows spanning two wines; wine 1 has a 3-row duplicate cell."""
    return pl.DataFrame(
        {
            "rating_id": [1, 2, 3, 4, 5, 6],
            "wine_id": [1, 1, 1, 1, 2, 2],
            "vintage_year": [2015, 2015, 2015, 2016, 2015, 2015],
            "age_at_review": [2, 2, 2, 1, 3, 4],
            "rating": [3.0, 4.0, 5.0, 4.5, 2.0, 3.0],
            "sample_weight": [0.5, 0.5, 0.5, 0.5, 1.5, 1.5],  # constant per wine
            "region_name": ["A", "A", "A", "A", "B", "B"],
            "gdd_10c": [1400.0, 1400.0, 1400.0, 1450.0, 1300.0, 1300.0],
        }
    )


def test_aggregate_rating_cells_means_ratings_and_sums_weights() -> None:
    cells = aggregate_rating_cells(_raw_rating_rows())

    # 6 raw rows → 4 unique (wine, vintage, age) cells, sorted by key.
    assert cells.height == 4
    dup = cells.filter(
        (pl.col("wine_id") == 1) & (pl.col("vintage_year") == 2015) & (pl.col("age_at_review") == 2)
    )
    assert dup.get_column("rating").item() == pytest.approx(4.0)  # mean of 3, 4, 5
    assert dup.get_column("sample_weight").item() == pytest.approx(1.5)  # 3 × 0.5
    assert dup.get_column(CELL_N_RATINGS_COL).item() == 3
    # Cell-constant features pass through unchanged.
    assert dup.get_column("gdd_10c").item() == pytest.approx(1400.0)
    # Singleton cells are identity rows.
    single = cells.filter(pl.col("wine_id") == 1).filter(pl.col("vintage_year") == 2016)
    assert single.get_column("rating").item() == pytest.approx(4.5)
    assert single.get_column(CELL_N_RATINGS_COL).item() == 1


def test_aggregate_rating_cells_is_deterministic() -> None:
    """Row order must be reproducible — CatBoost results depend on it."""
    df = _raw_rating_rows()
    shuffled = df.sample(fraction=1.0, shuffle=True, seed=7)
    assert aggregate_rating_cells(df).equals(aggregate_rating_cells(shuffled))


def test_aggregate_wine_vintage_drops_age_and_dedups() -> None:
    wv = aggregate_wine_vintage(_raw_rating_rows())

    assert "age_at_review" not in wv.columns  # wine-level labels can't depend on it
    assert wv.height == 3  # (1, 2015), (1, 2016), (2, 2015)
    w2 = wv.filter(pl.col("wine_id") == 2)
    assert w2.get_column(CELL_N_RATINGS_COL).item() == 2
    assert w2.get_column("sample_weight").item() == pytest.approx(3.0)


def test_cell_count_is_metadata_never_a_feature() -> None:
    """The collapse count is popularity — unknowable for unseen wines."""
    assert CELL_N_RATINGS_COL in MODEL_METADATA_COLS
    cols = [*_raw_rating_rows().columns, CELL_N_RATINGS_COL]
    spec = feature_spec(cols, targets=[RATING_TARGET])
    assert CELL_N_RATINGS_COL not in spec.feature_cols


def test_rating_cells_have_constant_features(processed_dataset) -> None:  # noqa: ANN001
    """Invariant behind the loss-exactness claim: no feature varies inside a cell.

    Aggregation takes `.first()` of every non-key column; if a future feature
    varied within (wine, vintage, age) that would silently corrupt training.
    Guarded here against the real build pipeline's output.
    """
    train = load_split("train")
    spec = feature_spec(train.columns, targets=[RATING_TARGET])
    # The keys themselves are features too, but they're constant by definition.
    checked = [c for c in spec.feature_cols if c not in RATING_CELL_KEYS]
    n_unique = (
        train.group_by(list(RATING_CELL_KEYS))
        .agg([pl.col(c).n_unique().alias(c) for c in checked])
        .select([pl.col(c).max() for c in checked])
    )
    varying = [c for c in checked if (n_unique.get_column(c).item() or 0) > 1]
    assert varying == [], f"features vary within a rating cell: {varying}"


def test_grouped_val_split_is_wine_disjoint() -> None:
    """No wine_id straddles the fit/validation boundary."""
    df = pl.DataFrame({"wine_id": sorted(list(range(1, 11)) * 2), "x": range(20)})
    fit, val = grouped_val_split(df, frac=0.3, seed=42)
    fit_ids = set(fit.get_column("wine_id").to_list())
    val_ids = set(val.get_column("wine_id").to_list())
    assert fit_ids.isdisjoint(val_ids)
    assert len(val_ids) == 3
    assert fit_ids | val_ids == set(range(1, 11))


def test_grouped_val_split_single_wine_yields_empty_val() -> None:
    """One wine cannot be split — caller trains without an eval set."""
    df = pl.DataFrame({"wine_id": [7, 7, 7], "x": [1, 2, 3]})
    fit, val = grouped_val_split(df, frac=0.3, seed=42)
    assert fit.height == 3
    assert val.is_empty()


def test_load_split_subsamples_by_wine(tmp_data_dir: Path) -> None:
    """sample_frac keeps a deterministic subset; full load keeps everything."""
    from vininator.config import get_settings

    settings = get_settings()
    df = pl.DataFrame({"wine_id": list(range(1000)), "rating": [3.5] * 1000})
    df.write_parquet(settings.processed_train_parquet)

    full = load_split("train")
    sub = load_split("train", sample_frac=0.2)
    assert full.height == 1000
    assert 0 < sub.height < full.height
    assert load_split("train", max_rows=10).height == 10
