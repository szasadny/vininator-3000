"""A thin MLflow wrapper so every trainer logs the same way.

Tracking is local and file-based (the gitignored repo-root `mlruns/`), needs no
server, and degrades gracefully: `track_run(..., enabled=False)` yields a
no-op logger, which is what the test suite and any "just train, don't record"
path use. The wrapper stamps each run with the git SHA and dataset hash so a
result can always be traced back to the code and data that produced it.
"""

from __future__ import annotations

import contextlib
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import mlflow

from vininator.config import MLFLOW_EXPERIMENT, get_settings


class RunLogger:
    """Logs metrics/params/artifacts to the active MLflow run, or nowhere.

    A single object is yielded by `track_run`; when tracking is disabled every
    method is a no-op, so trainer code calls it unconditionally.
    """

    def __init__(self, *, enabled: bool) -> None:
        self.enabled = enabled

    def log_params(self, params: dict[str, Any], *, prefix: str = "") -> None:
        if not self.enabled:
            return
        flat = {f"{prefix}{k}": _scalar(v) for k, v in params.items()}
        if flat:
            mlflow.log_params(flat)

    def log_metrics(self, metrics: dict[str, float], *, prefix: str = "") -> None:
        if not self.enabled:
            return
        clean = {
            f"{prefix}{k}": float(v) for k, v in metrics.items() if v is not None and _is_finite(v)
        }
        if clean:
            mlflow.log_metrics(clean)

    def log_artifact(self, path: Path) -> None:
        if not self.enabled:
            return
        mlflow.log_artifact(str(path))


def git_sha() -> str | None:
    """Current commit SHA, or `None` outside a git checkout."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip() or None
    except (subprocess.SubprocessError, OSError):
        return None


@contextlib.contextmanager
def track_run(
    run_name: str,
    *,
    params: dict[str, Any] | None = None,
    tags: dict[str, str] | None = None,
    dataset_hash: str | None = None,
    enabled: bool = True,
    tracking_uri: str | None = None,
    experiment: str = MLFLOW_EXPERIMENT,
) -> Iterator[RunLogger]:
    """Open an MLflow run (or a no-op) and yield a `RunLogger`.

    Args:
        run_name: Human-readable run name (e.g. ``"rating"``).
        params: Flat params logged at run start (config, feature count, …).
        tags: Extra MLflow tags.
        dataset_hash: Fingerprint of the training table, logged as a tag.
        enabled: When ``False`` nothing touches MLflow.
        tracking_uri: Override the local store (tests point this at tmp).
        experiment: MLflow experiment name.
    """
    if not enabled:
        yield RunLogger(enabled=False)
        return

    uri = tracking_uri or get_settings().mlflow_tracking_uri
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name):
        run_tags = {"git_sha": git_sha() or "unknown"}
        if dataset_hash is not None:
            run_tags["dataset_hash"] = dataset_hash
        if tags:
            run_tags.update(tags)
        mlflow.set_tags(run_tags)
        logger = RunLogger(enabled=True)
        if params:
            logger.log_params(params)
        yield logger


def _scalar(value: Any) -> Any:
    """Coerce a param value into something MLflow will accept."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _is_finite(value: float) -> bool:
    try:
        return float("-inf") < float(value) < float("inf")
    except (TypeError, ValueError):
        return False
