"""Profile models — Body and Acidity multi-class classifiers.

Two CatBoost classifiers over the same feature block as the rating model, one
per X-Wines structured label. Both labels are wine-level constants, so trainer
and eval collapse the rating rows to one row per (wine, vintage) first
(`dataset.aggregate_wine_vintage`, ~21× fewer rows on the full variant) —
per-rating rows would be pure duplication in training and would double-count
popular wines in the metrics. Both labels are heavily skewed (Acidity is ~83%
"High"), so we train with balanced class weights and report **macro-F1**
alongside accuracy — accuracy alone rewards always predicting the majority
class. Rows whose label is null are dropped (you cannot train a classifier on
a missing target); everything else mirrors `rating.py`.
"""

from __future__ import annotations

import dataclasses
import gc
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from catboost import CatBoostClassifier

from vininator.config import (
    ACIDITY_BUNDLE,
    BODY_BUNDLE,
    PROFILE_TARGETS,
    get_settings,
)
from vininator.eval.metrics import macro_f1
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
    load_split,
    load_train_config,
    notify,
    resolve_catboost_params,
    sample_weights,
)
from vininator.models.tracking import git_sha, track_run

_BUNDLE_BY_TARGET: dict[str, str] = {
    "body_label": BODY_BUNDLE,
    "acidity_label": ACIDITY_BUNDLE,
}


@dataclasses.dataclass(frozen=True)
class ProfileMetrics:
    """Accuracy + macro-F1 for one profile target on one eval split."""

    split: str
    target: str
    accuracy: float
    macro_f1: float


@dataclasses.dataclass(frozen=True)
class ProfileReport:
    """Summary returned by `train_profile`."""

    n_features: int
    metrics: list[ProfileMetrics]
    bundle_names: list[str]


def train_profile(
    config_path: Path,
    *,
    force: bool = False,  # noqa: ARG001 — symmetry with feature builders; training always rebuilds
    sample_frac: float | None = None,
    track: bool = True,
    notify_fn: NotifyFn | None = None,
) -> ProfileReport:
    """Train Body + Acidity classifiers and report accuracy + macro-F1.

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

    n_features = 0
    metrics: list[ProfileMetrics] = []
    names: list[str] = []

    with track_run("profile", dataset_hash=dhash, enabled=track) as logger:
        for target in PROFILE_TARGETS:
            notify(notify_fn, f"... fitting {target}")
            spec = feature_spec(train.columns, targets=[target])
            n_features = len(spec.feature_cols)

            tr = train.filter(pl.col(target).is_not_null())
            fit_df, val_df = grouped_val_split(tr, seed=cfg.seed)
            del tr

            train_pool = build_pool(
                fit_df,
                spec,
                label=fit_df.get_column(target).to_pandas(),
                weight=sample_weights(fit_df),
            )
            del fit_df
            gc.collect()

            eval_pool = (
                build_pool(
                    val_df,
                    spec,
                    label=val_df.get_column(target).to_pandas(),
                    weight=sample_weights(val_df),
                )
                if not val_df.is_empty()
                else None
            )
            del val_df
            gc.collect()

            model = _fit(cfg, target, train_pool, eval_pool)
            del train_pool, eval_pool
            gc.collect()

            target_metrics: list[ProfileMetrics] = []
            for split_name in ("test", "future_vintage_test"):
                notify(notify_fn, f"... evaluating {target} on {split_name}")
                eval_df = load_split(split_name, sample_frac=frac, max_rows=cfg.max_rows)
                # Eval per wine-vintage, mirroring training — per-rating eval
                # would let popular wines dominate accuracy and macro-F1.
                te = aggregate_wine_vintage(eval_df).filter(pl.col(target).is_not_null())
                del eval_df
                if te.is_empty():
                    del te
                    continue
                m = _evaluate(model, spec, te, target, split_name)
                del te
                target_metrics.append(m)
                logger.log_metrics(
                    {"accuracy": m.accuracy, "macro_f1": m.macro_f1},
                    prefix=f"{split_name}_{target}_",
                )

            metrics.extend(target_metrics)
            meta = _meta(cfg, spec, target, dhash, model, target_metrics)
            names.append(_save(model, meta, _BUNDLE_BY_TARGET[target], logger))

    return ProfileReport(n_features=n_features, metrics=metrics, bundle_names=names)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _fit(
    cfg: TrainConfig,
    target: str,
    train_pool: Any,
    eval_pool: Any | None,
) -> CatBoostClassifier:
    """Fit a balanced MultiClass classifier with early stopping by wine."""
    params = resolve_catboost_params(cfg, loss_function="MultiClass")
    params.setdefault("auto_class_weights", "Balanced")
    snapshot = get_settings().snapshots_dir / f"profile_{cfg.name}_{target}.cbm"
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
    return model


def _evaluate(
    model: CatBoostClassifier, spec: FeatureSpec, te: pl.DataFrame, target: str, split: str
) -> ProfileMetrics:
    """Accuracy + macro-F1 on the (label-present) rows of one eval split."""
    pool = build_pool(te, spec, label=None)
    preds = np.ravel(model.predict(pool)).astype(str)
    y = te.get_column(target).cast(pl.Utf8).to_numpy().astype(str)
    return ProfileMetrics(
        split=split,
        target=target,
        accuracy=float((y == preds).mean()),
        macro_f1=macro_f1(y, preds),
    )


def _meta(
    cfg: TrainConfig,
    spec: FeatureSpec,
    target: str,
    dhash: str,
    model: CatBoostClassifier,
    target_metrics: list[ProfileMetrics],
) -> dict[str, Any]:
    return {
        "model_class": "CatBoostClassifier",
        "target": target,
        "feature_cols": spec.feature_cols,
        "cat_cols": spec.cat_cols,
        "classes": [str(c) for c in model.classes_],
        "config_name": cfg.name,
        "dataset_hash": dhash,
        "git_sha": git_sha(),
        "metrics": {
            m.split: {"accuracy": m.accuracy, "macro_f1": m.macro_f1} for m in target_metrics
        },
    }


def _save(model: Any, meta: dict[str, Any], name: str, logger: Any) -> str:
    cbm_path, meta_path = save_bundle(model, meta, name=name)
    logger.log_artifact(cbm_path)
    logger.log_artifact(meta_path)
    return name
