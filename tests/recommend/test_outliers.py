"""Tests for recommend/outliers.py — peer baselines, the support gate, and the CI gate."""

from __future__ import annotations

import numpy as np

from vininator.config import Settings
from vininator.recommend.outliers import _OUTLIER_OUT_COLS, _peer_lookup, recommend_outliers


def test_peer_lookup_support_gate(processed_dataset: Settings) -> None:
    keys = ["region_name", "vintage_year"]
    # A permissive threshold keeps groups; an impossibly high one drops them all.
    assert _peer_lookup(keys, min_wines=1).height > 0
    assert _peer_lookup(keys, min_wines=10_000).is_empty()


def test_outlier_invariants(trained_bundles: Settings) -> None:
    report = recommend_outliers(opening_year=2026, min_peer_wines=1, top=10)

    assert report.path.exists()
    assert report.table.columns == list(_OUTLIER_OUT_COLS)

    if report.table.height:
        predicted = report.table.get_column("predicted_rating").to_numpy()
        baseline = report.table.get_column("peer_baseline").to_numpy()
        lo = report.table.get_column("predicted_rating_lo").to_numpy()
        over = report.table.get_column("overperformance").to_numpy()
        # The lower CI bound clears the baseline for every surviving outlier.
        assert (lo > baseline).all()
        # Overperformance is exactly predicted minus baseline.
        assert np.allclose(over, predicted - baseline)
        # Sorted by overperformance descending.
        assert over.tolist() == sorted(over.tolist(), reverse=True)


def test_support_gate_empties_result(trained_bundles: Settings) -> None:
    """An unreachable support threshold leaves no peer baseline, so nothing survives."""
    report = recommend_outliers(opening_year=2026, min_peer_wines=10_000, top=10)
    assert report.n_outliers == 0
    assert report.table.is_empty()
