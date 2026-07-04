"""Tests for eval/sanity.py — the qualitative known-wines check."""

from __future__ import annotations

from vininator.eval.sanity import sanity_check
from vininator.models.harmonize import train_harmonize
from vininator.models.profile import train_profile
from vininator.models.rating import train_rating


def test_sanity_check_scores_matched_wines(processed_dataset, write_config) -> None:
    train_rating(write_config("rating_cfg"), track=False)
    train_profile(write_config("profile_cfg"), track=False)
    train_harmonize(write_config("harmonize_cfg"), track=False)

    report = sanity_check(["Wine 1", "definitely-not-a-wine"], top_pairings=3)

    assert report.unmatched == ["definitely-not-a-wine"]
    assert report.rows, "expected at least one scored wine"
    row = report.rows[0]
    assert row.query == "Wine 1"
    assert row.split in {"train", "test", "future_vintage_test"}
    assert 1.0 <= row.predicted_rating <= 5.5
    assert row.predicted_body
    assert row.predicted_acidity
    assert len(row.predicted_pairings) == 3
    assert row.n_ratings >= 1
