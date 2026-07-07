"""Data assembly for RESULTS.md — pure functions the generator renders.

`scripts/build_results.py` orchestrates and formats; this module does the
computation so it stays testable. The only markdown decision here is
`markdown_table`; everything else returns polars frames or small dataclasses.
No file I/O beyond reading the trained bundles, processed splits, and the two
persisted eval parquets (ablations, SHAP).

Memory: the heavy functions project the split parquets to just the columns they
need (baselines need five columns, not all 127) and free each frame before the
next, mirroring the trainers' 16 GB discipline.
"""

from __future__ import annotations

import dataclasses
import gc
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.metrics import confusion_matrix, f1_score

from vininator.config import (
    ACIDITY_BUNDLE,
    BODY_BUNDLE,
    CELL_N_RATINGS_COL,
    HARMONIZE_BUNDLE,
    HARMONIZE_TARGET_PREFIX,
    RATING_BUNDLE,
    RATING_CELL_KEYS,
    RATING_QUANTILE_HI_BUNDLE,
    RATING_QUANTILE_LO_BUNDLE,
    RATING_TARGET,
    RESULTS_QUANTILE_NOMINAL,
    SAMPLE_WEIGHT_COL,
    SHAP_TOP_N_FEATURES,
    get_settings,
)
from vininator.eval.metrics import apply_rating_baselines, fit_rating_baselines, per_label_f1
from vininator.models.artifacts import load_bundle
from vininator.models.dataset import (
    SplitName,
    aggregate_rating_cells,
    aggregate_wine_vintage,
    build_pool,
    load_split,
    split_path,
)

_EVAL_SPLITS: tuple[SplitName, ...] = ("test", "future_vintage_test")
_BASELINE_ORDER = ("global_mean", "winery_mean", "region_vintage_mean", "grape_region_mean")
# Columns the leakage-safe baselines need — a projection this small turns the
# 15.5M-row train read from ~10 GB into a few hundred MB.
_BASELINE_COLS = (
    *RATING_CELL_KEYS,
    RATING_TARGET,
    SAMPLE_WEIGHT_COL,
    "winery_id",
    "region_name",
    "grape_majority",
)
_PROFILE_BUNDLE_BY_TARGET = {"body_label": BODY_BUNDLE, "acidity_label": ACIDITY_BUNDLE}


# ---------------------------------------------------------------------------
# Markdown formatting (the one place the script does not touch)
# ---------------------------------------------------------------------------


def write_markdown_atomic(target: Path, content: str) -> None:
    """Write text via a tmp sibling then rename; LF newlines even on Windows."""
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8", newline="\n")
    tmp.replace(target)


def markdown_table(df: pl.DataFrame, *, floats: str = "{:.4f}") -> str:
    """Render a small polars frame as a GitHub markdown table.

    Floats use `floats`; `None` becomes empty; list/tuple cells are joined with
    ", "; booleans render as `true`/`false`. Column names are the header.
    """

    def fmt(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, float):
            return floats.format(value)
        if isinstance(value, (list, tuple, pl.Series)):
            return ", ".join(str(item) for item in value)
        return str(value)

    header = "| " + " | ".join(df.columns) + " |"
    separator = "| " + " | ".join("---" for _ in df.columns) + " |"
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.iter_rows()]
    return "\n".join([header, separator, *body])


# ---------------------------------------------------------------------------
# Ranking display frames (shared by build_results.py and build_library.py)
# ---------------------------------------------------------------------------


def rating_band(table: pl.DataFrame) -> list[str]:
    """`"4.32 (4.10-4.55)"` per row — the prediction with its quantile band."""
    return [
        f"{p:.2f} ({lo:.2f}-{hi:.2f})"
        for p, lo, hi in zip(
            table.get_column("predicted_rating").to_list(),
            table.get_column("predicted_rating_lo").to_list(),
            table.get_column("predicted_rating_hi").to_list(),
            strict=True,
        )
    ]


def drink_now_display(table: pl.DataFrame) -> pl.DataFrame:
    """A drink-now / standout ranking as display columns for `markdown_table`."""
    return pl.DataFrame(
        {
            "Winery": table.get_column("winery_name").to_list(),
            "Wine": table.get_column("wine_name").to_list(),
            "Region": table.get_column("region_name").to_list(),
            "Vintage": table.get_column("vintage_year").to_list(),
            "Predicted (lo-hi)": rating_band(table),
            "Body": table.get_column("predicted_body").to_list(),
            "Acidity": table.get_column("predicted_acidity").to_list(),
            "Pairings": [", ".join(p) for p in table.get_column("top_pairings").to_list()],
        }
    )


def age_well_display(table: pl.DataFrame) -> pl.DataFrame:
    """An age-well summary ranking as display columns for `markdown_table`."""
    return pl.DataFrame(
        {
            "Winery": table.get_column("winery_name").to_list(),
            "Wine": table.get_column("wine_name").to_list(),
            "Region": table.get_column("region_name").to_list(),
            "Vintage": table.get_column("vintage_year").to_list(),
            "Peak yr": table.get_column("predicted_peak_year").to_list(),
            "Peak": [f"{x:.2f}" for x in table.get_column("predicted_peak_rating").to_list()],
            "Slope/yr": [f"{x:+.3f}" for x in table.get_column("slope_to_peak").to_list()],
            "Trajectory": table.get_column("trajectory").to_list(),
            "Clipped": table.get_column("age_clipped_any").to_list(),
        }
    )


def grape_display(slug: str) -> str:
    """`"cabernet-sauvignon"` → `"Cabernet Sauvignon"` for a section heading."""
    return slug.replace("-", " ").replace("_", " ").title()


def value_display(table: pl.DataFrame) -> pl.DataFrame:
    """A price/value ranking as display columns for `markdown_table`.

    Expects the scored+priced columns produced by `recommend.library.join_price`
    (`price_eur`, `price_band`, `match_confidence`) alongside the rating band.
    Rating-per-€10 is shown so the value column reads on the same 0-5 scale as
    the rating. Unpriced rows never reach here (the value views drop them).
    """
    price_eur = table.get_column("price_eur").to_list()
    predicted = table.get_column("predicted_rating").to_list()
    return pl.DataFrame(
        {
            "Winery": table.get_column("winery_name").to_list(),
            "Wine": table.get_column("wine_name").to_list(),
            "Region": table.get_column("region_name").to_list(),
            "Vintage": table.get_column("vintage_year").to_list(),
            "Predicted (lo-hi)": rating_band(table),
            "Price (EUR)": [f"€{p:.0f}" if p is not None else "" for p in price_eur],
            "Band": table.get_column("price_band").to_list(),
            "Rating/€10": [
                f"{r / p * 10:.2f}" if p else "" for r, p in zip(predicted, price_eur, strict=True)
            ],
            "Match": table.get_column("match_confidence").to_list(),
        }
    )


# ---------------------------------------------------------------------------
# Setup (§2)
# ---------------------------------------------------------------------------


def split_summary() -> pl.DataFrame:
    """Row / wine / wine-vintage counts and the vintage span per split.

    Lazy aggregation only — never materializes a full split just to count it.
    """
    records: list[dict[str, Any]] = []
    for split in ("train", *_EVAL_SPLITS):
        agg = (
            pl.scan_parquet(split_path(split))
            .select(
                pl.len().alias("ratings"),
                pl.col("wine_id").n_unique().alias("wines"),
                pl.struct("wine_id", "vintage_year").n_unique().alias("wine_vintages"),
                pl.col("vintage_year").min().alias("v_min"),
                pl.col("vintage_year").max().alias("v_max"),
            )
            .collect()
        )
        records.append(
            {
                "split": split,
                "ratings": int(agg["ratings"][0]),
                "wines": int(agg["wines"][0]),
                "wine_vintages": int(agg["wine_vintages"][0]),
                "vintages": f"{agg['v_min'][0]}-{agg['v_max'][0]}",
            }
        )
    return pl.DataFrame(records)


# ---------------------------------------------------------------------------
# Rating quality (§5)
# ---------------------------------------------------------------------------


def baseline_grid() -> dict[str, pl.DataFrame]:
    """Per split: the model row + leakage-safe baseline grid at both eval levels.

    Baselines are fit on the train fold once and applied to each eval split at
    per-rating and cell (weighted) level. The model row and noise floor are read
    from the rating bundle metadata — recomputing them here would just duplicate
    what training already logged.
    """
    rating_meta = load_bundle(RATING_BUNDLE).meta["metrics"]

    train = pl.scan_parquet(split_path("train")).select(_BASELINE_COLS).collect()
    fit = fit_rating_baselines(train)
    del train
    gc.collect()

    grids: dict[str, pl.DataFrame] = {}
    for split in _EVAL_SPLITS:
        proj = pl.scan_parquet(split_path(split)).select(_BASELINE_COLS).collect()
        per_rating = {b.name: b for b in apply_rating_baselines(fit, proj)}
        cells = aggregate_rating_cells(proj)
        per_cell = {
            b.name: b for b in apply_rating_baselines(fit, cells, weight_col=CELL_N_RATINGS_COL)
        }
        del proj, cells
        gc.collect()

        m = rating_meta[split]
        rows = [
            {
                "predictor": "model (CatBoost)",
                "rating_rmse": m["rmse"],
                "rating_mae": m["mae"],
                "cell_rmse": m["cell_rmse"],
                "cell_mae": m["cell_mae"],
            }
        ]
        rows += [
            {
                "predictor": name,
                "rating_rmse": per_rating[name].rmse,
                "rating_mae": per_rating[name].mae,
                "cell_rmse": per_cell[name].rmse,
                "cell_mae": per_cell[name].mae,
            }
            for name in _BASELINE_ORDER
        ]
        rows.append(
            {
                "predictor": "noise_floor",
                "rating_rmse": m["noise_floor"],
                "rating_mae": None,
                "cell_rmse": None,
                "cell_mae": None,
            }
        )
        grids[split] = pl.DataFrame(rows)
    return grids


def quantile_coverage() -> pl.DataFrame:
    """Per split: weighted coverage of observed cell means by the 0.1/0.9 band.

    The quantile heads band the wine-vintage mean rating, so coverage is checked
    on aggregated cells weighted by ratings per cell — the same level the heads
    were trained and the recommender consumes them at.
    """
    q_lo = load_bundle(RATING_QUANTILE_LO_BUNDLE)
    q_hi = load_bundle(RATING_QUANTILE_HI_BUNDLE)
    spec = q_lo.feature_spec()

    rows: list[dict[str, Any]] = []
    for split in _EVAL_SPLITS:
        cells = aggregate_rating_cells(load_split(split))
        pool = build_pool(cells, spec, label=None)
        lo = np.asarray(q_lo.model.predict(pool))
        hi = np.asarray(q_hi.model.predict(pool))
        observed = cells.get_column(RATING_TARGET).to_numpy()
        weight = cells.get_column(CELL_N_RATINGS_COL).to_numpy().astype(float)
        covered = (observed >= lo) & (observed <= hi)
        rows.append(
            {
                "split": split,
                "nominal": RESULTS_QUANTILE_NOMINAL,
                "coverage": float(np.average(covered, weights=weight)),
                "mean_band_width": float(np.average(hi - lo, weights=weight)),
                "n_cells": cells.height,
            }
        )
        del cells, pool
        gc.collect()
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# Profile + Harmonize quality (§6)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ProfileEvalReport:
    """One profile target's headline metrics, per-class F1, and confusion matrix."""

    target: str
    classes: list[str]
    headline: pl.DataFrame  # split, accuracy, macro_f1 (both splits, from meta)
    per_class_f1: pl.DataFrame  # class, f1 (test split)
    confusion: pl.DataFrame  # rows = actual, cols = predicted (test split)


@dataclasses.dataclass(frozen=True)
class HarmonizeEvalReport:
    """Harmonize headline metrics, per-label F1, and a few example wines."""

    headline: pl.DataFrame  # split, mean_f1, hamming (both splits, from meta)
    per_label_f1: pl.DataFrame  # pairing, f1 (test split, sorted desc)
    examples: pl.DataFrame  # wine, region, vintage, predicted, actual pairings


def load_test_wine_vintages() -> pl.DataFrame:
    """The test split collapsed to one row per (wine, vintage) — the profile grain."""
    return aggregate_wine_vintage(load_split("test"))


def profile_eval(target: str, *, wv: pl.DataFrame | None = None) -> ProfileEvalReport:
    """Per-class F1 + confusion matrix for a Body/Acidity classifier on the test split.

    Headline accuracy / macro-F1 for both splits come from the bundle metadata;
    the per-class detail is recomputed on the (label-present) test wine-vintages.
    """
    bundle = load_bundle(_PROFILE_BUNDLE_BY_TARGET[target])
    classes = [str(c) for c in bundle.meta["classes"]]
    meta_metrics = bundle.meta["metrics"]
    headline = pl.DataFrame(
        [
            {
                "split": s,
                "accuracy": meta_metrics[s]["accuracy"],
                "macro_f1": meta_metrics[s]["macro_f1"],
            }
            for s in meta_metrics
        ]
    )

    if wv is None:
        wv = load_test_wine_vintages()
    te = wv.filter(pl.col(target).is_not_null())
    preds = np.ravel(
        bundle.model.predict(build_pool(te, bundle.feature_spec(), label=None))
    ).astype(str)
    actual = te.get_column(target).cast(pl.Utf8).to_numpy().astype(str)

    f1s = f1_score(actual, preds, labels=classes, average=None, zero_division=0)
    per_class_f1 = pl.DataFrame({"class": classes, "f1": [float(x) for x in f1s]})

    matrix = confusion_matrix(actual, preds, labels=classes)
    confusion = pl.DataFrame(
        {
            "actual \\ predicted": classes,
            **{c: matrix[:, j].tolist() for j, c in enumerate(classes)},
        }
    )
    return ProfileEvalReport(
        target=target,
        classes=classes,
        headline=headline,
        per_class_f1=per_class_f1,
        confusion=confusion,
    )


def harmonize_eval(*, wv: pl.DataFrame | None = None) -> HarmonizeEvalReport:
    """Per-label F1 + example pairings for the harmonize model on the test split.

    Mean-F1 / Hamming for both splits come from the bundle metadata; per-label F1
    (not persisted) is recomputed on the test wine-vintages with the frozen
    per-label thresholds, and three high-coverage wines illustrate the output.
    """
    bundle = load_bundle(HARMONIZE_BUNDLE)
    labels = list(bundle.meta["targets"])
    thresholds = bundle.meta["thresholds"]
    meta_metrics = bundle.meta["metrics"]
    headline = pl.DataFrame(
        [
            {
                "split": s,
                "mean_f1": meta_metrics[s]["mean_f1"],
                "hamming": meta_metrics[s]["hamming"],
            }
            for s in meta_metrics
        ]
    )

    if wv is None:
        wv = load_test_wine_vintages()
    proba = np.asarray(
        bundle.model.predict_proba(build_pool(wv, bundle.feature_spec(), label=None))
    )
    thr = np.array([thresholds[label] for label in labels])
    y_pred = (proba >= thr).astype(int)
    y_true = wv.select(labels).to_numpy()

    stripped = [_strip_pair(label) for label in labels]
    f1_by = per_label_f1(y_true, y_pred, labels)
    per_label = pl.DataFrame({"pairing": stripped, "f1": [f1_by[label] for label in labels]}).sort(
        "f1", descending=True
    )

    examples = _harmonize_examples(wv, y_pred, y_true, stripped)
    return HarmonizeEvalReport(headline=headline, per_label_f1=per_label, examples=examples)


def _harmonize_examples(
    wv: pl.DataFrame, y_pred: np.ndarray, y_true: np.ndarray, stripped: list[str]
) -> pl.DataFrame:
    """Three most-rated wine-vintages with predicted vs. actual pairing sets."""
    counts = (
        wv.get_column(CELL_N_RATINGS_COL).to_numpy()
        if CELL_N_RATINGS_COL in wv.columns
        else np.arange(wv.height)
    )
    order = [int(i) for i in np.argsort(counts)[::-1][:3]]

    names = _wine_name_lookup(wv.get_column("wine_id").to_list())
    wine_ids = wv.get_column("wine_id").to_list()
    regions = wv.get_column("region_name").to_list()
    vintages = wv.get_column("vintage_year").to_list()

    rows: list[dict[str, Any]] = []
    for i in order:
        predicted = [stripped[j] for j in range(len(stripped)) if y_pred[i, j] == 1]
        actual = [stripped[j] for j in range(len(stripped)) if y_true[i, j] == 1]
        rows.append(
            {
                "wine": names.get(wine_ids[i], str(wine_ids[i])),
                "region": regions[i],
                "vintage": vintages[i],
                "predicted": predicted,
                "actual": actual,
            }
        )
    return pl.DataFrame(rows)


def _wine_name_lookup(wine_ids: list[int]) -> dict[int, str]:
    """`wine_id` → `WineName` for the given ids, from the raw wines parquet."""
    frame = (
        pl.scan_parquet(get_settings().xwines_wines_parquet)
        .filter(pl.col("WineID").is_in(wine_ids))
        .select("WineID", "WineName")
        .collect()
    )
    return dict(
        zip(
            frame.get_column("WineID").to_list(),
            frame.get_column("WineName").to_list(),
            strict=True,
        )
    )


# ---------------------------------------------------------------------------
# Diagnostics (§7)
# ---------------------------------------------------------------------------


def ablation_table() -> pl.DataFrame:
    """The persisted ablation grid as one row per arm, cell-RMSE + Δ vs. full."""
    df = pl.read_parquet(get_settings().ablations_parquet)
    test = {r["arm"]: r for r in df.filter(pl.col("split") == "test").iter_rows(named=True)}
    fv = {
        r["arm"]: r
        for r in df.filter(pl.col("split") == "future_vintage_test").iter_rows(named=True)
    }
    full_test = test["full"]["cell_rmse"]
    full_fv = fv["full"]["cell_rmse"] if "full" in fv else None

    rows: list[dict[str, Any]] = []
    for arm in ("full", "no_terroir", "no_producer", "no_age"):
        if arm not in test:
            continue
        t = test[arm]["cell_rmse"]
        row: dict[str, Any] = {
            "arm": arm,
            "n_features": int(test[arm]["n_features"]),
            "test_cell_rmse": t,
            "test_delta": t - full_test,
        }
        if arm in fv and full_fv is not None:
            f = fv[arm]["cell_rmse"]
            row["fv_cell_rmse"] = f
            row["fv_delta"] = f - full_fv
        else:
            row["fv_cell_rmse"] = None
            row["fv_delta"] = None
        rows.append(row)
    return pl.DataFrame(rows)


def shap_table(top_n: int = SHAP_TOP_N_FEATURES) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Top-`top_n` features by mean |SHAP| and the per-block rollup."""
    df = pl.read_parquet(get_settings().shap_importance_parquet)
    top = (
        df.sort("mean_abs_shap", descending=True)
        .head(top_n)
        .select("feature", "block", "mean_abs_shap")
    )
    block = (
        df.group_by("block")
        .agg(pl.col("mean_abs_shap").sum().alias("total_mean_abs_shap"))
        .sort("total_mean_abs_shap", descending=True)
    )
    return top, block


# ---------------------------------------------------------------------------
# Limitations (§9)
# ---------------------------------------------------------------------------


def readme_disclaimer() -> str:
    """Extract the README's `## Disclaimer` section so §9 stays in sync with it."""
    root = get_settings().results_md.parent
    text = (root / "README.md").read_text(encoding="utf-8")
    marker = "\n## Disclaimer\n"
    start = text.find(marker)
    if start == -1:
        raise ValueError("README.md '## Disclaimer' heading not found; update readme_disclaimer().")
    body_start = start + len(marker)
    end = text.find("\n## ", body_start)
    section = text[body_start:end] if end != -1 else text[body_start:]
    return section.strip()


def _strip_pair(column: str) -> str:
    """`pair_beef` → `beef` for display."""
    return column.removeprefix(HARMONIZE_TARGET_PREFIX)
