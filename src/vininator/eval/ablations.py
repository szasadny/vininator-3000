"""Feature-block ablations for the rating model (Phase 5).

Retrains the RMSE head — and only the RMSE head; the quantile heads don't
change the ablation conclusion and would triple the cost — once per arm with
one feature block removed, then evaluates every arm on the same two held-out
splits the trainer uses. The `full` arm retrains with nothing removed so every
delta is measured against a reference fit on identical data with an identical
early-stopping fold, not against a bundle trained in some other run.

Arms:
  full         — reference, no columns dropped
  no_terroir   — drop the climate + soil block (`terroir_feature_cols()`)
  no_producer  — drop the leave-one-wine-out producer aggregates
  no_age       — drop `age_at_review`

Memory shape (the 16 GB constraint): the raw train split is loaded and
cell-aggregated exactly once, and the cells are parked in a scratch parquet
under `snapshots_dir`. Each arm then re-scans that parquet with the arm's
columns projected out — polars projection pushdown means the dropped block is
never even read — so no arm ever holds the raw 15.5M-row frame alongside a
CatBoost pool. The scratch parquet is deleted when the run finishes.

Each arm is logged to MLflow as `rating_ablation_<arm>`, and the full grid is
written to `settings.ablations_parquet` for RESULTS.md generation.
"""

from __future__ import annotations

import dataclasses
import gc
from pathlib import Path

import polars as pl

from vininator.config import (
    AGE_COL,
    PRODUCER_FEATURE_COLS,
    RATING_TARGET,
    get_settings,
)
from vininator.features.terroir import terroir_feature_cols
from vininator.models.dataset import (
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
from vininator.models.rating import EvalMetrics, evaluate_regressor, fit_regressor
from vininator.models.tracking import track_run

_EVAL_SPLITS = ("test", "future_vintage_test")
_SCRATCH_CELLS_STEM = "ablation_cells"


@dataclasses.dataclass(frozen=True)
class AblationResult:
    """One arm's retrained-model metrics on both eval splits."""

    arm: str
    dropped_cols: tuple[str, ...]
    n_features: int
    metrics: list[EvalMetrics]


@dataclasses.dataclass(frozen=True)
class AblationReport:
    """Summary returned by `run_ablations`."""

    results: list[AblationResult]
    parquet_path: Path


def ablation_arms() -> dict[str, tuple[str, ...]]:
    """Arm name → feature columns dropped for that arm, in report order."""
    return {
        "full": (),
        "no_terroir": tuple(terroir_feature_cols()),
        "no_producer": PRODUCER_FEATURE_COLS,
        "no_age": (AGE_COL,),
    }


def run_ablations(
    config_path: Path,
    *,
    arms: list[str] | None = None,
    sample_frac: float | None = None,
    track: bool = True,
    notify_fn: NotifyFn | None = None,
) -> AblationReport:
    """Retrain the RMSE head per arm and report cell-level deltas.

    Args:
        config_path: The same experiment yaml the trainer uses — arms must be
            trained with identical hyperparameters or the deltas are noise.
        arms: Subset of `ablation_arms()` to run (default: all, `full` first).
        sample_frac: Overrides the config's `data.sample_frac` (smoke runs).
        track: Log each arm to MLflow as `rating_ablation_<arm>`.
        notify_fn: Milestone callback (CLI passes `typer.echo`).

    Returns:
        An `AblationReport`; the grid parquet is written even for a subset run
        (overwriting any previous grid — arms are meaningless across configs).
    """
    settings = get_settings()
    settings.ensure_dirs()
    cfg = load_train_config(config_path)
    frac = sample_frac if sample_frac is not None else cfg.sample_frac

    all_arms = ablation_arms()
    selected = list(all_arms) if arms is None else arms
    unknown = [a for a in selected if a not in all_arms]
    if unknown:
        raise ValueError(f"Unknown ablation arm(s) {unknown}; expected {list(all_arms)}")

    notify(notify_fn, "... loading + aggregating train split (once, shared by all arms)")
    raw_train = load_split("train", sample_frac=frac, max_rows=cfg.max_rows)
    dhash = dataset_hash(raw_train)
    cells = aggregate_rating_cells(raw_train)
    del raw_train
    gc.collect()

    cells_path = settings.snapshots_dir / f"{_SCRATCH_CELLS_STEM}_{cfg.name}.parquet"
    cells.write_parquet(cells_path)
    n_cells = cells.height
    del cells
    gc.collect()
    notify(notify_fn, f"... {n_cells:,} train cells parked at {cells_path.name}")

    try:
        results = [
            _run_arm(cfg, arm, all_arms[arm], cells_path, frac, dhash, track, notify_fn)
            for arm in selected
        ]
    finally:
        cells_path.unlink(missing_ok=True)

    parquet_path = _write_grid(results, settings.ablations_parquet)
    return AblationReport(results=results, parquet_path=parquet_path)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _run_arm(
    cfg: TrainConfig,
    arm: str,
    dropped: tuple[str, ...],
    cells_path: Path,
    frac: float | None,
    dhash: str,
    track: bool,
    notify_fn: NotifyFn | None,
) -> AblationResult:
    """Train one arm from the shared cells parquet and evaluate both splits."""
    schema_names = pl.scan_parquet(cells_path).collect_schema().names()
    keep = [c for c in schema_names if c not in dropped]
    # Projection pushdown: the dropped block never leaves the parquet.
    cells = pl.scan_parquet(cells_path).select(keep).collect()
    spec = feature_spec(cells.columns, targets=[RATING_TARGET])
    notify(notify_fn, f"... [{arm}] {len(spec.feature_cols)} features (dropped {len(dropped)})")

    fit_df, val_df = grouped_val_split(cells, seed=cfg.seed)
    del cells
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
        "arm": arm,
        "n_dropped": len(dropped),
        "n_features": len(spec.feature_cols),
        "sample_frac": frac,
        **resolve_catboost_params(cfg, loss_function="RMSE"),
    }
    with track_run(
        f"rating_ablation_{arm}", params=run_params, dataset_hash=dhash, enabled=track
    ) as logger:
        notify(notify_fn, f"... [{arm}] fitting RMSE head")
        model = fit_regressor(
            cfg, "RMSE", train_pool, eval_pool, snapshot_stem=f"ablation_{arm}_{cfg.name}"
        )
        del train_pool, eval_pool
        gc.collect()

        metrics: list[EvalMetrics] = []
        for split_name in _EVAL_SPLITS:
            eval_df = load_split(split_name, sample_frac=frac, max_rows=cfg.max_rows)
            if eval_df.is_empty():
                del eval_df
                continue
            m = evaluate_regressor(model, spec, eval_df, split_name)
            metrics.append(m)
            logger.log_metrics(
                {
                    "rmse": m.rmse,
                    "mae": m.mae,
                    "cell_rmse": m.cell_rmse,
                    "cell_mae": m.cell_mae,
                    "noise_floor": m.noise_floor,
                },
                prefix=f"{split_name}_",
            )
            del eval_df
            gc.collect()

    del model
    gc.collect()
    return AblationResult(
        arm=arm, dropped_cols=dropped, n_features=len(spec.feature_cols), metrics=metrics
    )


def _write_grid(results: list[AblationResult], target: Path) -> Path:
    """Flatten results to one row per (arm, split) and write atomically."""
    rows = [
        {
            "arm": r.arm,
            "split": m.split,
            "n_features": r.n_features,
            "n_dropped": len(r.dropped_cols),
            "rmse": m.rmse,
            "mae": m.mae,
            "cell_rmse": m.cell_rmse,
            "cell_mae": m.cell_mae,
            "noise_floor": m.noise_floor,
        }
        for r in results
        for m in r.metrics
    ]
    df = pl.DataFrame(rows)
    tmp = target.with_suffix(target.suffix + ".tmp")
    df.write_parquet(tmp)
    tmp.replace(target)
    return target
