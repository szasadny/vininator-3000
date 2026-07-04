"""Rating regressor — the headline Phase 4 model.

A CatBoost RMSE regressor over the full feature block, trained on aggregated
(wine, vintage, age) feature cells — loss-exact vs. the raw rating rows but
~7× smaller (see `dataset.aggregate_rating_cells`). Two extra quantile heads
(alpha 0.1 / 0.9) band the *cell-mean* rating and become the `_lo` / `_hi`
columns the Phase 6 recommender reads. `age_at_review` is an ordinary numeric
feature, which is what lets the recommender sweep opening years at inference.

Evaluation happens at two levels on both held-out splits, next to the
leakage-safe baselines:

- **per-rating** RMSE/MAE — comparable 1:1 to the PROJECT.md baseline numbers,
  but bounded below by the within-cell spread of user opinions (the logged
  `noise_floor`, ~0.64 on the full variant vs. a global std of ~0.74);
- **cell-level** RMSE/MAE, weighted by ratings per cell — predicted vs.
  observed mean rating per (wine, vintage, age). This is the headline metric:
  it measures the wine-quality question the recommender actually asks, with
  the user-opinion noise averaged out, so the terroir ablation delta is
  visible instead of drowned.
"""

from __future__ import annotations

import dataclasses
import gc
from pathlib import Path
from typing import Any

import polars as pl
from catboost import CatBoostRegressor

from vininator.config import (
    CELL_N_RATINGS_COL,
    RATING_BUNDLE,
    RATING_CELL_KEYS,
    RATING_QUANTILE_HI_BUNDLE,
    RATING_QUANTILE_LO_BUNDLE,
    RATING_QUANTILES,
    RATING_TARGET,
    get_settings,
)
from vininator.eval.metrics import (
    BaselineResult,
    apply_rating_baselines,
    fit_rating_baselines,
    mae,
    rmse,
    within_group_std,
)
from vininator.models.artifacts import save_bundle
from vininator.models.dataset import (
    FeatureSpec,
    NotifyFn,
    TrainConfig,
    aggregate_rating_cells,
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


@dataclasses.dataclass(frozen=True)
class EvalMetrics:
    """Model metrics on one eval split, at both evaluation levels.

    `rmse`/`mae` are per-rating (comparable to the baseline literature but
    floored by `noise_floor`, the within-cell spread of user opinions);
    `cell_rmse`/`cell_mae` are per (wine, vintage, age) cell, weighted by
    ratings per cell — the headline numbers.
    """

    split: str
    rmse: float
    mae: float
    cell_rmse: float
    cell_mae: float
    noise_floor: float


@dataclasses.dataclass(frozen=True)
class RatingReport:
    """Summary returned by `train_rating`.

    `baselines` is per-rating, `cell_baselines` the same grid at cell level —
    each keyed by split name.
    """

    n_train: int
    n_features: int
    eval_metrics: list[EvalMetrics]
    baselines: dict[str, list[BaselineResult]]
    cell_baselines: dict[str, list[BaselineResult]]
    bundle_names: list[str]


def train_rating(
    config_path: Path,
    *,
    force: bool = False,  # noqa: ARG001 — symmetry with feature builders; training always rebuilds
    sample_frac: float | None = None,
    track: bool = True,
    notify_fn: NotifyFn | None = None,
) -> RatingReport:
    """Train the rating regressor + quantile heads and report against baselines.

    Args:
        config_path: Experiment yaml (CatBoost params + data subset).
        force: Accepted for CLI symmetry; training always retrains.
        sample_frac: Overrides the config's `data.sample_frac` (smoke runs).
        track: Log to MLflow. `False` for tests / quick local runs.
        notify_fn: Milestone callback (CLI passes `typer.echo`).

    Returns:
        A `RatingReport` with model metrics and the baseline grid.
    """
    settings = get_settings()
    settings.ensure_dirs()
    cfg = load_train_config(config_path)
    frac = sample_frac if sample_frac is not None else cfg.sample_frac

    notify(notify_fn, "... loading train split")
    raw_train = load_split("train", sample_frac=frac, max_rows=cfg.max_rows)
    n_source_rows = raw_train.height
    dhash = dataset_hash(raw_train)

    # Fit baseline lookups while we still have the raw frame, then collapse it
    # to feature cells — the lookups are tiny (KBs) and survive the whole run;
    # the multi-GB raw frame does not.
    baseline_fit = fit_rating_baselines(raw_train)
    train = aggregate_rating_cells(raw_train)
    del raw_train
    gc.collect()

    spec = feature_spec(train.columns, targets=[RATING_TARGET])
    n_train = train.height
    notify(
        notify_fn,
        f"... {len(spec.feature_cols)} features, {n_source_rows:,} ratings -> {n_train:,} cells",
    )

    fit_df, val_df = grouped_val_split(train, seed=cfg.seed)
    del train
    gc.collect()

    train_pool = build_pool(
        fit_df,
        spec,
        label=fit_df.get_column(RATING_TARGET).to_pandas(),
        weight=sample_weights(fit_df),
    )
    del fit_df
    gc.collect()

    eval_pool = (
        build_pool(
            val_df,
            spec,
            label=val_df.get_column(RATING_TARGET).to_pandas(),
            weight=sample_weights(val_df),
        )
        if not val_df.is_empty()
        else None
    )
    del val_df
    gc.collect()

    run_params = {
        "config": cfg.name,
        "n_features": len(spec.feature_cols),
        "n_train_cells": n_train,
        "n_source_rows": n_source_rows,
        "sample_frac": frac,
        **resolve_catboost_params(cfg, loss_function="RMSE"),
    }

    with track_run("rating", params=run_params, dataset_hash=dhash, enabled=track) as logger:
        notify(notify_fn, "... fitting RMSE regressor")
        model = fit_regressor(cfg, "RMSE", train_pool, eval_pool)

        notify(notify_fn, "... fitting quantile heads")
        lo_alpha, hi_alpha = RATING_QUANTILES
        q_lo = fit_regressor(cfg, f"Quantile:alpha={lo_alpha}", train_pool, eval_pool)
        q_hi = fit_regressor(cfg, f"Quantile:alpha={hi_alpha}", train_pool, eval_pool)

        # Pools are no longer needed — free before loading eval splits.
        del train_pool, eval_pool
        gc.collect()

        eval_metrics: list[EvalMetrics] = []
        baselines: dict[str, list[BaselineResult]] = {}
        cell_baselines: dict[str, list[BaselineResult]] = {}
        for split_name in ("test", "future_vintage_test"):
            notify(notify_fn, f"... evaluating on {split_name}")
            eval_df = load_split(split_name, sample_frac=frac, max_rows=cfg.max_rows)
            if eval_df.is_empty():
                del eval_df
                continue
            metrics = evaluate_regressor(model, spec, eval_df, split_name)
            eval_metrics.append(metrics)
            baselines[split_name] = apply_rating_baselines(baseline_fit, eval_df)
            cells = aggregate_rating_cells(eval_df)
            cell_baselines[split_name] = apply_rating_baselines(
                baseline_fit, cells, weight_col=CELL_N_RATINGS_COL
            )
            del cells
            logger.log_metrics(
                {
                    "rmse": metrics.rmse,
                    "mae": metrics.mae,
                    "cell_rmse": metrics.cell_rmse,
                    "cell_mae": metrics.cell_mae,
                    "noise_floor": metrics.noise_floor,
                },
                prefix=f"{split_name}_",
            )
            for b in baselines[split_name]:
                logger.log_metrics({f"{b.name}_rmse": b.rmse}, prefix=f"{split_name}_")
            for b in cell_baselines[split_name]:
                logger.log_metrics({f"cell_{b.name}_rmse": b.rmse}, prefix=f"{split_name}_")
            del eval_df
            gc.collect()

        meta = _base_meta(cfg, spec, dhash, eval_metrics)
        names = [
            _save(model, {**meta, "bundle": RATING_BUNDLE}, RATING_BUNDLE, logger),
            _save(
                q_lo,
                {**meta, "bundle": RATING_QUANTILE_LO_BUNDLE, "alpha": lo_alpha},
                RATING_QUANTILE_LO_BUNDLE,
                logger,
            ),
            _save(
                q_hi,
                {**meta, "bundle": RATING_QUANTILE_HI_BUNDLE, "alpha": hi_alpha},
                RATING_QUANTILE_HI_BUNDLE,
                logger,
            ),
        ]

    return RatingReport(
        n_train=n_train,
        n_features=len(spec.feature_cols),
        eval_metrics=eval_metrics,
        baselines=baselines,
        cell_baselines=cell_baselines,
        bundle_names=names,
    )


# ---------------------------------------------------------------------------
# Fit + evaluate helpers (shared with eval/ablations.py)
# ---------------------------------------------------------------------------


def fit_regressor(
    cfg: TrainConfig,
    loss_function: str,
    train_pool: Any,
    eval_pool: Any | None,
    *,
    snapshot_stem: str | None = None,
) -> CatBoostRegressor:
    """Fit one CatBoostRegressor with the given loss, early-stopping if possible.

    `snapshot_stem` namespaces the crash-recovery snapshot file so concurrent
    or interleaved trainings (e.g. the ablation arms) never resume from each
    other's snapshots.
    """
    params = resolve_catboost_params(cfg, loss_function=loss_function)
    # Snapshot every 5 min so a crash can resume rather than restart from scratch.
    # Requires `allow_writing_files: true` in the config (the resolver defaults
    # it off for tests); CatBoost's scratch logs then land in snapshots_dir
    # instead of a `catboost_info/` dir in the CWD.
    safe_loss = loss_function.split(":")[0]
    stem = snapshot_stem if snapshot_stem is not None else f"rating_{cfg.name}"
    snapshot = get_settings().snapshots_dir / f"{stem}_{safe_loss}.cbm"
    use_snapshot = params.get("allow_writing_files", True) is not False
    if use_snapshot:
        params.setdefault("train_dir", str(get_settings().snapshots_dir))
    model = CatBoostRegressor(**params)
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


def evaluate_regressor(
    model: CatBoostRegressor, spec: FeatureSpec, eval_df: pl.DataFrame, split_name: str
) -> EvalMetrics:
    """Per-rating and cell-level RMSE/MAE (plus the noise floor) on one eval split.

    Cell predictions reuse the per-rating predictions: features are constant
    within a cell, so predicting once per raw row and averaging per cell gives
    the exact per-cell prediction without building a second pool.
    """
    pool = build_pool(eval_df, spec, label=None)
    preds = model.predict(pool)
    y = eval_df.get_column(RATING_TARGET).to_numpy()

    keys = list(RATING_CELL_KEYS)
    cells = (
        eval_df.select([*keys, RATING_TARGET])
        .with_columns(pl.Series("_pred", preds))
        .group_by(keys)
        .agg(pl.col(RATING_TARGET).mean(), pl.col("_pred").first(), pl.len().alias("_n"))
    )
    return EvalMetrics(
        split=split_name,
        rmse=rmse(y, preds),
        mae=mae(y, preds),
        cell_rmse=rmse(
            cells.get_column(RATING_TARGET), cells.get_column("_pred"), cells.get_column("_n")
        ),
        cell_mae=mae(
            cells.get_column(RATING_TARGET), cells.get_column("_pred"), cells.get_column("_n")
        ),
        noise_floor=within_group_std(eval_df, RATING_CELL_KEYS),
    )


def _base_meta(
    cfg: TrainConfig,
    spec: FeatureSpec,
    dhash: str,
    eval_metrics: list[EvalMetrics],
) -> dict[str, Any]:
    return {
        "model_class": "CatBoostRegressor",
        "target": RATING_TARGET,
        "feature_cols": spec.feature_cols,
        "cat_cols": spec.cat_cols,
        "config_name": cfg.name,
        "dataset_hash": dhash,
        "git_sha": git_sha(),
        "quantiles": list(RATING_QUANTILES),
        "metrics": {
            m.split: {
                "rmse": m.rmse,
                "mae": m.mae,
                "cell_rmse": m.cell_rmse,
                "cell_mae": m.cell_mae,
                "noise_floor": m.noise_floor,
            }
            for m in eval_metrics
        },
    }


def _save(model: Any, meta: dict[str, Any], name: str, logger: Any) -> str:
    cbm_path, meta_path = save_bundle(model, meta, name=name)
    logger.log_artifact(cbm_path)
    logger.log_artifact(meta_path)
    return name
