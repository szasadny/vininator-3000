"""Fixtures for the Phase 6 recommender tests.

`processed_dataset` and `write_config` are shared from tests/models/conftest.py
(the recommender scores the same synthetic processed parquets the trainers
build). `trained_bundles` trains all six bundles on that synthetic data with a
tiny config so the scoring path has real models to load.
"""

from __future__ import annotations

import pytest

from tests.models.conftest import processed_dataset, write_config  # noqa: F401
from vininator.config import Settings
from vininator.models.harmonize import train_harmonize
from vininator.models.profile import train_profile
from vininator.models.rating import train_rating


@pytest.fixture
def trained_bundles(processed_dataset: Settings, write_config) -> Settings:  # noqa: F811
    """Train rating (+ quantile heads), profile, and harmonize bundles.

    Returns the settings singleton so tests can resolve model / output paths.
    """
    train_rating(write_config("rating_cfg"), track=False)
    train_profile(write_config("profile_cfg"), track=False)
    train_harmonize(write_config("harmonize_cfg"), track=False)
    return processed_dataset
