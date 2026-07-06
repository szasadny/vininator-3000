"""Tests for eval/sanity.py — the qualitative known-wines check."""

from __future__ import annotations

from vininator.config import Settings
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


def test_preferred_vintage_is_honored_when_present(trained_bundles: Settings) -> None:
    # Wine 3 gets a synthetic 2020 (future-vintage) rating; pin it explicitly.
    report = sanity_check(["Wine 3"], preferred_vintages={"Wine 3": 2020}, max_matches=1)
    assert len(report.rows) == 1
    assert report.rows[0].vintage_year == 2020


def test_preferred_vintage_falls_back_when_absent(trained_bundles: Settings) -> None:
    # Wine 1 has no 2099 vintage → fall back to its most-rated (2015-2017).
    report = sanity_check(["Wine 1"], preferred_vintages={"Wine 1": 2099}, max_matches=1)
    assert len(report.rows) == 1
    assert report.rows[0].vintage_year in {2015, 2016, 2017}


def test_max_matches_caps_per_query(trained_bundles: Settings) -> None:
    # "Wine" matches every synthetic wine; the cap keeps exactly one.
    report = sanity_check(["Wine"], max_matches=1)
    assert len(report.rows) == 1
