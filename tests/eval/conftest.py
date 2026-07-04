"""Fixture re-exports for the eval tests.

The Phase 5 eval modules exercise the same processed parquets and trained
bundles as the model trainers, so the synthetic-dataset and tiny-config
fixtures are shared from tests/models/conftest.py rather than duplicated.
"""

from __future__ import annotations

from tests.models.conftest import processed_dataset, write_config  # noqa: F401
