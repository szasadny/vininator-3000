"""SHAP feature importance for the rating model (Phase 5).

Uses CatBoost's native `ShapValues` (exact TreeSHAP, no `shap` package
dependency) on a seeded sample of held-out cells. Two artifacts feed
RESULTS.md §7:

- `data/processed/shap_importance.parquet` — mean |SHAP| per feature, tagged
  with the feature block (terroir / producer / age / wine) so the block-level
  rollup that answers "does terroir contribute above producer + region +
  grape" is one group-by away.
- `reports/figures/shap_*.png` — the top-N summary bar chart and the
  dependence scatters for the terroir variables in
  `config.SHAP_DEPENDENCE_FEATURES`.

Figures are static matplotlib (Agg backend), styled to the project's report
conventions: a single series hue, hairline gridlines, text in ink tokens —
the data is the only loud thing on the canvas.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from vininator.config import (
    AGE_COL,
    MODEL_SEED,
    PRODUCER_FEATURE_COLS,
    RATING_BUNDLE,
    SHAP_DEPENDENCE_FEATURES,
    SHAP_SAMPLE_CELLS,
    SHAP_TOP_N_FEATURES,
    get_settings,
)
from vininator.features.terroir import terroir_feature_cols
from vininator.models.artifacts import load_bundle
from vininator.models.dataset import (
    NotifyFn,
    SplitName,
    aggregate_rating_cells,
    build_pool,
    load_split,
    notify,
)

# Report-figure tokens (light surface; see .claude notes on chart conventions).
_SURFACE = "#fcfcfb"
_INK = "#0b0b0b"
_INK_SECONDARY = "#52514e"
_INK_MUTED = "#898781"
_GRIDLINE = "#e1e0d9"
_BASELINE = "#c3c2b7"
_SERIES = "#2a78d6"


@dataclasses.dataclass(frozen=True)
class ShapReport:
    """Summary returned by `run_shap_analysis`."""

    split: str
    n_cells: int
    importance_path: Path
    figure_paths: list[Path]
    top_features: list[tuple[str, float]]
    block_totals: dict[str, float]


def run_shap_analysis(
    *,
    split: SplitName = "test",
    sample_cells: int | None = None,
    bundle_name: str = RATING_BUNDLE,
    notify_fn: NotifyFn | None = None,
) -> ShapReport:
    """Compute SHAP values on sampled eval cells; write the parquet + figures.

    Args:
        split: Which held-out split to explain (`test` by default — unseen
            wines are the honest view of what the model uses).
        sample_cells: Cells sampled (seeded) from the aggregated split;
            `None` uses `config.SHAP_SAMPLE_CELLS`.
        bundle_name: Saved bundle to explain (the RMSE rating head).
        notify_fn: Milestone callback (CLI passes `typer.echo`).

    Returns:
        A `ShapReport` with artifact paths and the ranking preview.
    """
    settings = get_settings()
    if sample_cells is None:
        sample_cells = SHAP_SAMPLE_CELLS
    bundle = load_bundle(bundle_name)
    spec = bundle.feature_spec()

    notify(notify_fn, f"... loading {split} split and aggregating cells")
    cells = aggregate_rating_cells(load_split(split))
    if cells.height > sample_cells:
        cells = cells.sample(n=sample_cells, seed=MODEL_SEED)
    notify(notify_fn, f"... computing SHAP values on {cells.height:,} cells")
    pool = build_pool(cells, spec, label=None)
    # Shape (n, n_features + 1); the trailing column is the expected value.
    shap = np.asarray(bundle.model.get_feature_importance(type="ShapValues", data=pool))
    contributions = shap[:, :-1]

    mean_abs = np.abs(contributions).mean(axis=0)
    importance = pl.DataFrame(
        {
            "feature": spec.feature_cols,
            "block": [_feature_block(c) for c in spec.feature_cols],
            "mean_abs_shap": mean_abs,
        }
    ).sort("mean_abs_shap", descending=True)

    importance_path = _write_atomic(importance, settings.shap_importance_parquet)
    block_totals = {
        row["block"]: row["total"]
        for row in importance.group_by("block")
        .agg(pl.col("mean_abs_shap").sum().alias("total"))
        .sort("total", descending=True)
        .iter_rows(named=True)
    }

    notify(notify_fn, "... rendering figures")
    settings.figures_dir.mkdir(parents=True, exist_ok=True)
    figures = [_plot_summary(importance, split, settings.figures_dir)]
    for feature in SHAP_DEPENDENCE_FEATURES:
        if feature not in spec.feature_cols:
            notify(notify_fn, f"... skipping dependence plot for absent feature {feature!r}")
            continue
        idx = spec.feature_cols.index(feature)
        figures.append(
            _plot_dependence(
                feature,
                cells.get_column(feature),
                contributions[:, idx],
                split,
                settings.figures_dir,
            )
        )

    top = importance.head(SHAP_TOP_N_FEATURES)
    return ShapReport(
        split=split,
        n_cells=cells.height,
        importance_path=importance_path,
        figure_paths=figures,
        top_features=list(
            zip(
                top.get_column("feature").to_list(),
                top.get_column("mean_abs_shap").to_list(),
                strict=True,
            )
        ),
        block_totals=block_totals,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _feature_block(feature: str) -> str:
    """Tag a feature with its ablation block (terroir / producer / age / wine)."""
    if feature in set(terroir_feature_cols()):
        return "terroir"
    if feature in PRODUCER_FEATURE_COLS:
        return "producer"
    if feature == AGE_COL:
        return "age"
    return "wine"


def _write_atomic(df: pl.DataFrame, target: Path) -> Path:
    tmp = target.with_suffix(target.suffix + ".tmp")
    df.write_parquet(tmp)
    tmp.replace(target)
    return target


def _style_axes(ax: plt.Axes) -> None:
    """Recessive chrome: hairline grid, muted ticks, no box."""
    ax.set_facecolor(_SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_BASELINE)
        ax.spines[side].set_linewidth(1)
    ax.tick_params(colors=_INK_MUTED, labelsize=8, length=0)


def _plot_summary(importance: pl.DataFrame, split: str, out_dir: Path) -> Path:
    """Horizontal top-N mean-|SHAP| bars, largest on top, values at the tips."""
    top = importance.head(SHAP_TOP_N_FEATURES).reverse()
    features = top.get_column("feature").to_list()
    values = top.get_column("mean_abs_shap").to_list()

    fig, ax = plt.subplots(figsize=(7.5, 0.32 * len(features) + 1.0), dpi=150)
    fig.patch.set_facecolor(_SURFACE)
    _style_axes(ax)
    ax.xaxis.grid(True, color=_GRIDLINE, linewidth=1)
    ax.set_axisbelow(True)

    ax.barh(features, values, height=0.62, color=_SERIES)
    for i, v in enumerate(values):
        ax.text(
            v + max(values) * 0.01,
            i,
            f"{v:.3f}",
            va="center",
            fontsize=7,
            color=_INK_SECONDARY,
        )
    ax.set_xlim(0, max(values) * 1.12)
    ax.set_title(
        f"Rating model — mean |SHAP| per feature, top {len(features)} ({split} cells)",
        loc="left",
        fontsize=10,
        color=_INK,
    )
    ax.set_xlabel("mean |SHAP| (rating points)", fontsize=8, color=_INK_SECONDARY)

    path = out_dir / f"shap_summary_{split}.png"
    fig.tight_layout()
    fig.savefig(path, facecolor=_SURFACE)
    plt.close(fig)
    return path


def _plot_dependence(
    feature: str,
    values: pl.Series,
    shap_values: np.ndarray,
    split: str,
    out_dir: Path,
) -> Path:
    """Feature value vs SHAP contribution scatter for one terroir variable."""
    x = values.cast(pl.Float64).to_numpy()
    keep = ~np.isnan(x)
    x, y = x[keep], shap_values[keep]

    fig, ax = plt.subplots(figsize=(6.0, 4.0), dpi=150)
    fig.patch.set_facecolor(_SURFACE)
    _style_axes(ax)
    ax.yaxis.grid(True, color=_GRIDLINE, linewidth=1)
    ax.set_axisbelow(True)
    ax.axhline(0, color=_BASELINE, linewidth=1)

    is_boolean = values.dtype == pl.Boolean
    if is_boolean:
        # Jitter the two columns so density is visible; label the levels.
        rng = np.random.default_rng(MODEL_SEED)
        ax.scatter(
            x + rng.uniform(-0.08, 0.08, size=x.shape),
            y,
            s=6,
            alpha=0.2,
            color=_SERIES,
            edgecolors="none",
        )
        ax.set_xticks([0.0, 1.0], ["false", "true"])
    else:
        ax.scatter(x, y, s=6, alpha=0.2, color=_SERIES, edgecolors="none")

    ax.set_title(
        f"SHAP dependence — {feature} ({split} cells)", loc="left", fontsize=10, color=_INK
    )
    ax.set_xlabel(feature, fontsize=8, color=_INK_SECONDARY)
    ax.set_ylabel("SHAP value (rating points)", fontsize=8, color=_INK_SECONDARY)

    path = out_dir / f"shap_dependence_{feature}_{split}.png"
    fig.tight_layout()
    fig.savefig(path, facecolor=_SURFACE)
    plt.close(fig)
    return path
