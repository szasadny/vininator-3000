"""Harmonize model — multilabel food-pairing classifier.

One CatBoost `MultiLogloss` model predicts the whole `pair_*` vector jointly
(top-30 food pairings). It is the closest stand-in for a tasting profile the
text-free X-Wines dataset allows: "pairs with grilled red meat and hard cheese"
is a structural claim about body and intensity.

The pairing vector is a wine-level constant, so trainer, threshold tuning, and
eval all collapse the rating rows to one row per (wine, vintage) first
(`dataset.aggregate_wine_vintage`) — see `profile.py` for the rationale.

Per-label decision thresholds are tuned on the wine-grouped validation fold by
maximizing F1 (a flat 0.5 cut is wrong for rare pairings), then frozen into the
bundle so inference is deterministic. We report per-label F1 and Hamming loss
on held-out wine-vintages.
"""

from __future__ import annotations

import dataclasses
import gc
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from catboost import CatBoostClassifier

from vininator.config import HARMONIZE_BUNDLE, get_settings
from vininator.eval.metrics import hamming, per_label_f1
from vininator.models.artifacts import save_bundle
from vininator.models.dataset import (
    FeatureSpec,
    NotifyFn,
    TrainConfig,
    aggregate_wine_vintage,
    build_pool,
    dataset_hash,
    feature_spec,
    grouped_val_split,
    harmonize_target_cols,
    load_split,
    load_train_config,
    notify,
    resolve_catboost_params,
    sample_weights,
)
from vininator.models.tracking import git_sha, track_run

# Probability cuts tried per label when tuning thresholds.
_THRESHOLD_GRID = np.round(np.arange(0.05, 0.96, 0.05), 2)


@dataclasses.dataclass(frozen=True)
class HarmonizeSplitMetrics:
    """Per-label F1 + Hamming on one eval split."""

    split: str
    mean_f1: float
    hamming: float
    per_label_f1: dict[str, float]


@dataclasses.dataclass(frozen=True)
class HarmonizeReport:
    """Summary returned by `train_harmonize`."""

    n_features: int
    n_labels: int
    metrics: list[HarmonizeSplitMetrics]
    bundle_names: list[str]


def train_harmonize(
    config_path: Path,
    *,
    force: bool = False,  # noqa: ARG001 — symmetry with feature builders; training always rebuilds
    sample_frac: float | None = None,
    track: bool = True,
    notify_fn: NotifyFn | None = None,
) -> HarmonizeReport:
    """Train the multilabel food-pairing model and report per-label F1 + Hamming.

    Args:
        config_path: Experiment yaml (CatBoost params + data subset).
        force: Accepted for CLI symmetry; training always retrains.
        sample_frac: Overrides the config's `data.sample_frac` (smoke runs).
        track: Log to MLflow. `False` for tests / quick local runs.
        notify_fn: Milestone callback (CLI passes `typer.echo`).
    """
    settings = get_settings()
    settings.ensure_dirs()
    cfg = load_train_config(config_path)
    frac = sample_frac if sample_frac is not None else cfg.sample_frac

    notify(notify_fn, "... loading train split")
    raw_train = load_split("train", sample_frac=frac, max_rows=cfg.max_rows)
    dhash = dataset_hash(raw_train)
    train = aggregate_wine_vintage(raw_train)
    notify(notify_fn, f"... {raw_train.height:,} ratings -> {train.height:,} wine-vintages")
    del raw_train
    gc.collect()

    labels = harmonize_target_cols(train.columns)
    spec = feature_spec(train.columns, targets=labels)

    # val_df is retained after the fit for threshold tuning; train is not needed
    # beyond the pool-build phase.
    fit_df, val_df = grouped_val_split(train, seed=cfg.seed)
    del train
    gc.collect()

    train_pool = build_pool(
        fit_df,
        spec,
        label=fit_df.select(labels).to_pandas(),
        weight=sample_weights(fit_df),
    )
    del fit_df
    gc.collect()

    eval_pool = (
        build_pool(
            val_df,
            spec,
            label=val_df.select(labels).to_pandas(),
            weight=sample_weights(val_df),
        )
        if not val_df.is_empty()
        else None
    )

    with track_run("harmonize", dataset_hash=dhash, enabled=track) as logger:
        notify(notify_fn, f"... fitting MultiLogloss over {len(labels)} pairings")
        params = resolve_catboost_params(cfg, loss_function="MultiLogloss")
        snapshot = get_settings().snapshots_dir / f"harmonize_{cfg.name}.cbm"
        use_snapshot = params.get("allow_writing_files", True) is not False
        if use_snapshot:
            params.setdefault("train_dir", str(get_settings().snapshots_dir))
        model = CatBoostClassifier(**params)
        model.fit(
            train_pool,
            eval_set=eval_pool,
            use_best_model=eval_pool is not None,
            verbose=100,
            save_snapshot=use_snapshot,
            snapshot_file=str(snapshot) if use_snapshot else None,
            snapshot_interval=300 if use_snapshot else None,
        )
        if use_snapshot:
            snapshot.unlink(missing_ok=True)

        del train_pool, eval_pool
        gc.collect()

        notify(notify_fn, "... tuning per-label thresholds")
        thresholds = _tune_thresholds(model, spec, val_df, labels)
        del val_df
        gc.collect()

        split_metrics: list[HarmonizeSplitMetrics] = []
        for split_name in ("test", "future_vintage_test"):
            notify(notify_fn, f"... evaluating on {split_name}")
            raw_eval = load_split(split_name, sample_frac=frac, max_rows=cfg.max_rows)
            if raw_eval.is_empty():
                del raw_eval
                continue
            # Eval per wine-vintage, mirroring training — per-rating eval
            # would let popular wines dominate the per-label F1.
            eval_df = aggregate_wine_vintage(raw_eval)
            del raw_eval
            sm = _evaluate(model, spec, eval_df, labels, thresholds, split_name)
            del eval_df
            gc.collect()
            split_metrics.append(sm)
            logger.log_metrics(
                {"mean_f1": sm.mean_f1, "hamming": sm.hamming}, prefix=f"{split_name}_"
            )

        meta = _meta(cfg, spec, labels, thresholds, dhash, split_metrics)
        names = [_save(model, meta, HARMONIZE_BUNDLE, logger)]

    return HarmonizeReport(
        n_features=len(spec.feature_cols),
        n_labels=len(labels),
        metrics=split_metrics,
        bundle_names=names,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _tune_thresholds(
    model: CatBoostClassifier,
    spec: FeatureSpec,
    val_df: pl.DataFrame,
    labels: list[str],
) -> dict[str, float]:
    """Pick the F1-maximizing probability cut per label on the val fold.

    Falls back to 0.5 for every label when there is no validation data.
    """
    if val_df.is_empty():
        return dict.fromkeys(labels, 0.5)
    proba = _predict_proba(model, spec, val_df)
    y = val_df.select(labels).to_numpy()
    thresholds: dict[str, float] = {}
    for i, label in enumerate(labels):
        best_t, best_f1 = 0.5, -1.0
        for t in _THRESHOLD_GRID:
            f1 = _binary_f1(y[:, i], (proba[:, i] >= t).astype(int))
            if f1 > best_f1:
                best_t, best_f1 = float(t), f1
        thresholds[label] = best_t
    return thresholds


def _evaluate(
    model: CatBoostClassifier,
    spec: FeatureSpec,
    eval_df: pl.DataFrame,
    labels: list[str],
    thresholds: dict[str, float],
    split: str,
) -> HarmonizeSplitMetrics:
    """Apply tuned thresholds on one eval split → per-label F1 + Hamming."""
    proba = _predict_proba(model, spec, eval_df)
    thr = np.array([thresholds[label] for label in labels])
    y_pred = (proba >= thr).astype(int)
    y_true = eval_df.select(labels).to_numpy()
    f1_by_label = per_label_f1(y_true, y_pred, labels)
    return HarmonizeSplitMetrics(
        split=split,
        mean_f1=float(np.mean(list(f1_by_label.values()))),
        hamming=hamming(y_true, y_pred),
        per_label_f1=f1_by_label,
    )


def _predict_proba(model: CatBoostClassifier, spec: FeatureSpec, df: pl.DataFrame) -> np.ndarray:
    """Per-label probabilities, shape (n_rows, n_labels)."""
    pool = build_pool(df, spec, label=None)
    return np.asarray(model.predict_proba(pool))


def _binary_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """F1 for a single binary label without a sklearn round-trip per cut."""
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    denom = 2 * tp + fp + fn
    return (2 * tp) / denom if denom else 0.0


def _meta(
    cfg: TrainConfig,
    spec: FeatureSpec,
    labels: list[str],
    thresholds: dict[str, float],
    dhash: str,
    split_metrics: list[HarmonizeSplitMetrics],
) -> dict[str, Any]:
    return {
        "model_class": "CatBoostClassifier",
        "targets": labels,
        "feature_cols": spec.feature_cols,
        "cat_cols": spec.cat_cols,
        "thresholds": thresholds,
        "config_name": cfg.name,
        "dataset_hash": dhash,
        "git_sha": git_sha(),
        "metrics": {
            sm.split: {"mean_f1": sm.mean_f1, "hamming": sm.hamming} for sm in split_metrics
        },
    }


def _save(model: Any, meta: dict[str, Any], name: str, logger: Any) -> str:
    cbm_path, meta_path = save_bundle(model, meta, name=name)
    logger.log_artifact(cbm_path)
    logger.log_artifact(meta_path)
    return name
