"""Tests for eval/metrics.py — metric math and leakage-safe baselines.

The baseline canary is the important one: a baseline must fit its lookup on the
**train** frame and never peek at the eval frame's ratings. If it leaked, the
RMSE would collapse toward zero and the headline terroir comparison would be
meaningless.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from vininator.eval.metrics import (
    apply_rating_baselines,
    baseline_global_mean,
    baseline_region_vintage_mean,
    fit_rating_baselines,
    hamming,
    macro_f1,
    mae,
    per_label_f1,
    rating_baselines,
    rmse,
    within_group_std,
)


def test_rmse_and_mae_exact() -> None:
    assert rmse([1, 2, 3], [1, 2, 3]) == pytest.approx(0.0)
    assert rmse([0, 0], [1, 1]) == pytest.approx(1.0)
    assert mae([0, 0, 0], [1, 2, 3]) == pytest.approx(2.0)


def test_weighted_rmse() -> None:
    # Weight the second (larger) error twice as heavily.
    val = rmse([0, 0], [1, 3], weight=[1.0, 2.0])
    assert val == pytest.approx(np.sqrt((1 * 1 + 2 * 9) / 3))


def test_macro_f1_perfect_and_known() -> None:
    assert macro_f1([0, 1, 0, 1], [0, 1, 0, 1]) == pytest.approx(1.0)


def test_per_label_f1_and_hamming() -> None:
    y_true = np.array([[1, 0], [0, 1], [1, 1]])
    y_pred = np.array([[1, 0], [0, 0], [1, 1]])
    f1 = per_label_f1(y_true, y_pred, ["a", "b"])
    assert f1["a"] == pytest.approx(1.0)  # column a predicted perfectly
    assert 0.0 <= f1["b"] <= 1.0
    assert hamming(y_true, y_pred) == pytest.approx(1 / 6)  # one wrong cell of six


def test_global_mean_baseline_fits_on_train_only() -> None:
    """Canary: predicts the TRAIN mean, not the eval mean."""
    train = pl.DataFrame({"rating": [4.0, 4.0, 4.0, 4.0]})
    eval_df = pl.DataFrame({"rating": [2.0, 2.0]})
    result = baseline_global_mean(train, eval_df)
    # Predicting train mean 4.0 against eval 2.0 → error 2.0. Leakage would give 0.
    assert result.rmse == pytest.approx(2.0)


def test_region_vintage_baseline_uses_group_mean_with_fallback() -> None:
    train = pl.DataFrame(
        {
            "region_name": ["A", "A", "B", "B"],
            "vintage_year": [2015, 2015, 2016, 2016],
            "rating": [4.0, 4.0, 3.0, 3.0],
        }
    )
    eval_df = pl.DataFrame(
        {
            "region_name": ["A", "Z"],  # Z is unseen → global-mean fallback
            "vintage_year": [2015, 2099],
            "rating": [4.0, 3.5],  # 3.5 == train global mean
        }
    )
    result = baseline_region_vintage_mean(train, eval_df)
    assert result.rmse == pytest.approx(0.0)  # group hit + fallback both land exactly


def test_within_group_std_pools_across_groups() -> None:
    """The noise floor is the pooled within-group std; single-row groups are skipped."""
    df = pl.DataFrame(
        {
            "wine_id": [1, 1, 2, 2, 3],  # wine 3 has one row → no spread information
            "rating": [3.0, 5.0, 4.0, 4.0, 1.0],
        }
    )
    # Wine 1: var 2.0 (dof 1); wine 2: var 0.0 (dof 1) → pooled std = sqrt(1.0).
    assert within_group_std(df, ["wine_id"]) == pytest.approx(1.0)


def test_within_group_std_no_spread_information_is_zero() -> None:
    df = pl.DataFrame({"wine_id": [1, 2, 3], "rating": [1.0, 3.0, 5.0]})
    assert within_group_std(df, ["wine_id"]) == pytest.approx(0.0)


def test_apply_rating_baselines_weighted_matches_row_expansion() -> None:
    """Cell-level baselines weighted by cell size must equal the per-rating numbers."""
    train = pl.DataFrame(
        {
            "region_name": ["A", "A", "B"],
            "vintage_year": [2015, 2015, 2016],
            "grape_majority": ["Merlot", "Merlot", "Syrah"],
            "winery_id": [1, 1, 2],
            "rating": [4.0, 4.0, 3.0],
        }
    )
    raw = pl.DataFrame(
        {
            "region_name": ["A", "A", "A", "B"],
            "vintage_year": [2015, 2015, 2015, 2016],
            "grape_majority": ["Merlot", "Merlot", "Merlot", "Syrah"],
            "winery_id": [1, 1, 1, 2],
            "rating": [5.0, 5.0, 5.0, 2.0],
        }
    )
    cells = pl.DataFrame(
        {
            "region_name": ["A", "B"],
            "vintage_year": [2015, 2016],
            "grape_majority": ["Merlot", "Syrah"],
            "winery_id": [1, 2],
            "rating": [5.0, 2.0],  # cell means of the raw rows above
            "n": [3, 1],
        }
    )
    fit = fit_rating_baselines(train)
    per_rating = apply_rating_baselines(fit, raw)
    per_cell = apply_rating_baselines(fit, cells, weight_col="n")
    for a, b in zip(per_rating, per_cell, strict=True):
        assert a.name == b.name
        assert a.rmse == pytest.approx(b.rmse)
        assert a.mae == pytest.approx(b.mae)


def test_rating_baselines_returns_full_grid() -> None:
    train = pl.DataFrame(
        {
            "region_name": ["A", "A"],
            "vintage_year": [2015, 2016],
            "grape_majority": ["Merlot", "Merlot"],
            "winery_id": [1, 2],
            "rating": [4.0, 3.0],
        }
    )
    names = [b.name for b in rating_baselines(train, train)]
    assert names == ["global_mean", "winery_mean", "region_vintage_mean", "grape_region_mean"]
