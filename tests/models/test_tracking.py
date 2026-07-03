"""Tests for models/tracking.py — the MLflow wrapper degrades gracefully."""

from __future__ import annotations

from pathlib import Path

from vininator.models.tracking import git_sha, track_run


def test_disabled_run_is_a_noop() -> None:
    """enabled=False yields a logger whose methods do nothing (and never raise)."""
    with track_run("t", params={"a": 1}, enabled=False) as logger:
        assert logger.enabled is False
        logger.log_params({"x": 1})
        logger.log_metrics({"rmse": 0.5})
        logger.log_artifact(Path("nonexistent"))


def test_enabled_run_writes_to_local_store(tmp_path: Path) -> None:
    """A real run materializes the MLflow file store under the given uri."""
    store = tmp_path / "mlruns"
    with track_run(
        "t",
        params={"a": 1},
        dataset_hash="abc",
        enabled=True,
        tracking_uri=store.as_uri(),
    ) as logger:
        logger.log_metrics({"rmse": 0.5})
    assert store.exists()


def test_git_sha_is_str_or_none() -> None:
    sha = git_sha()
    assert sha is None or isinstance(sha, str)
