"""Typer CLI entrypoint.

`vininator` is the single orchestration surface — phases (`data`, `features`,
`train`, and later `recommend`) hang off subcommand groups, each added without
touching this file's structure.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import typer

from vininator.data.geocode import (
    filter_to_usable,
    geocode_regions,
    result_type_distribution,
    scan_geocode,
)
from vininator.data.load import download_xwines, xwines_info
from vininator.eval.ablations import run_ablations
from vininator.eval.importance import run_shap_analysis
from vininator.eval.sanity import sanity_check
from vininator.features.build import build_processed_tables
from vininator.features.climate import build_climate_table
from vininator.features.soil import build_soil_table
from vininator.features.terroir import build_terroir_table
from vininator.models.harmonize import HarmonizeReport, train_harmonize
from vininator.models.profile import ProfileReport, train_profile
from vininator.models.rating import RatingReport, train_rating

app = typer.Typer(
    name="vininator",
    help="Vininator 3000 — wine rating and tasting-notes predictor.",
    no_args_is_help=True,
)

data_app = typer.Typer(help="Dataset acquisition and inspection.", no_args_is_help=True)
app.add_typer(data_app, name="data")

features_app = typer.Typer(help="Feature engineering and terroir pipeline.", no_args_is_help=True)
app.add_typer(features_app, name="features")

train_app = typer.Typer(
    help="Train the rating, profile, and harmonize models.", no_args_is_help=True
)
app.add_typer(train_app, name="train")

eval_app = typer.Typer(
    help="Phase 5 evaluation: ablations, SHAP, qualitative sanity checks.",
    no_args_is_help=True,
)
app.add_typer(eval_app, name="eval")


@data_app.command("download")
def data_download(
    force: bool = typer.Option(
        False, "--force", help="Re-fetch and re-normalize even if the parquets already exist."
    ),
) -> None:
    """Fetch the X-Wines variant's CSVs (test = auto, slim/full = manual drop)."""
    paths = download_xwines(force=force)
    for name, path in paths.items():
        typer.echo(f"X-Wines {name:8s} parquet: {path}")


@data_app.command("info")
def data_info() -> None:
    """Print row counts, schemas, and missingness for both X-Wines parquets."""
    typer.echo(json.dumps(xwines_info(), indent=2, default=str))


@features_app.command("geocode")
def features_geocode(
    force: bool = typer.Option(
        False, "--force", help="Discard the existing geocode cache and re-fetch everything."
    ),
    limit: int | None = typer.Option(
        None,
        "--limit",
        help="Process at most N regions this run (resumable — re-run to cover the rest).",
    ),
) -> None:
    """Geocode unique RegionName values to (lat, lon) via Nominatim.

    Rate-limited to 1 req/sec per Nominatim's TOS. ~2,160 regions on the full
    variant => ~35 minutes for a cold run; subsequent runs are no-ops. Progress
    is printed at every checkpoint flush; Ctrl+C is safe — partial results are
    persisted before the process exits.
    """

    def progress(done: int, total: int) -> None:
        pct = 100.0 * done / total if total else 100.0
        typer.echo(f"... geocoded {done}/{total} ({pct:.1f}%)")

    path = geocode_regions(force=force, limit=limit, progress_fn=progress, notify_fn=typer.echo)
    typer.echo(f"Geocode cache: {path}")


@features_app.command("geocode-audit")
def features_geocode_audit() -> None:
    """Print the `result_type` distribution from `geocode.parquet`.

    Helps tune the blacklist in `config.GEOCODE_BAD_RESULT_TYPES` against the
    real data — Nominatim tags wine regions with a wildly varied set of
    `result_type` values, so a tight whitelist drops real regions.
    """
    import unicodedata

    def _ascii(s: str | None) -> str:
        """Fold to ASCII so Windows cp1252 consoles don't crash on accents."""
        if s is None:
            return ""
        return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")

    df = scan_geocode().collect()
    n_total = df.height
    n_ok = df.filter(pl.col("status") == "ok").height
    n_kept = filter_to_usable(df).height
    typer.echo(f"total rows:           {n_total}")
    typer.echo(f"status='ok':          {n_ok}")
    typer.echo(f"after blacklist:      {n_kept}  (drops {n_ok - n_kept} rows)")
    typer.echo("")
    typer.echo(f"{'result_type':25s} {'n':>5s}  {'blacklisted':>11s}  example_region")
    typer.echo("-" * 80)
    for row in result_type_distribution(df).iter_rows(named=True):
        rt = row["result_type"] if row["result_type"] is not None else "(null)"
        mark = "yes" if row["blacklisted"] else ""
        typer.echo(f"{rt:25s} {row['n']:>5d}  {mark:>11s}  {_ascii(row['example_region'])}")


@features_app.command("soil")
def features_soil(
    force: bool = typer.Option(
        False, "--force", help="Discard the existing soil cache and re-fetch everything."
    ),
    limit: int | None = typer.Option(
        None, "--limit", help="Process at most N regions this run (resumable)."
    ),
) -> None:
    """Pull SoilGrids 0-30cm profile + Open-Elevation DEM for every geocoded region.

    Reads `data/interim/geocode.parquet` (status='ok' rows only). Each region
    costs ~9 SoilGrids + 1 Open-Elevation HTTP calls; results land in
    `data/interim/soil.parquet`. Resumable: re-runs only process missing rows.
    """

    def progress(done: int, total: int) -> None:
        pct = 100.0 * done / total if total else 100.0
        typer.echo(f"... soil {done}/{total} ({pct:.1f}%)")

    path = build_soil_table(force=force, limit=limit, progress_fn=progress, notify_fn=typer.echo)
    typer.echo(f"Soil table: {path}")


@features_app.command("climate")
def features_climate(
    force: bool = typer.Option(False, "--force", help="Re-fetch every JSON and rebuild every row."),
    limit: int | None = typer.Option(
        None, "--limit", help="Process at most N regions this run (resumable)."
    ),
) -> None:
    """Pull NASA POWER daily climate per region and derive climate features.

    Reads `data/interim/geocode.parquet` (status='ok' rows surviving the
    `filter_to_usable` blacklist). One HTTP request per region covers all years
    1991–2021 across the five daily variables; the JSON cache is the resume
    state. Smoke test: `vininator features climate --limit 5` — expect
    seconds to a minute total on a cold cache.
    """

    def progress(done: int, total: int) -> None:
        pct = 100.0 * done / total if total else 100.0
        typer.echo(f"... climate {done}/{total} ({pct:.1f}%)")

    path = build_climate_table(force=force, limit=limit, progress_fn=progress, notify_fn=typer.echo)
    typer.echo(f"Climate table: {path}")


@features_app.command("terroir")
def features_terroir(
    force: bool = typer.Option(
        False,
        "--force",
        help="Rebuild from current inputs (the build always rebuilds; flag is accepted for CLI symmetry).",
    ),
) -> None:
    """Join climate.parquet ⨝ soil.parquet → terroir.parquet.

    Pure compose: left-joins soil onto climate on (region, country). No
    network, no incremental state — running it twice gives the same parquet
    as long as the inputs haven't changed.
    """
    path = build_terroir_table(force=force, notify_fn=typer.echo)
    typer.echo(f"Terroir table: {path}")


@features_app.command("build")
def features_build(
    force: bool = typer.Option(
        False,
        "--force",
        help="Overwrite existing processed parquets even if they already exist.",
    ),
) -> None:
    """Assemble the final modeling table from X-Wines + terroir.

    Joins wines × ratings × terroir (one row per rating), adds grape and
    food-pairing multi-hot features, computes producer aggregates and sample
    weights on the training fold only, then writes three leakage-safe parquets
    to data/processed/:

    \b
      train.parquet              ~85% of wines, vintage ≤ 2018
      test.parquet               ~15% of wines, vintage ≤ 2018
      future_vintage_test.parquet  all wines, vintage 2019–2021

    Requires geocode.parquet, climate.parquet, soil.parquet, and terroir.parquet
    to exist (run the earlier `features` sub-commands first).

    NV ratings (null Vintage) are silently dropped — they cannot join terroir.
    """
    report = build_processed_tables(force=force, notify_fn=typer.echo)
    typer.echo(f"train rows:                {report.train_rows:>12,}")
    typer.echo(f"test rows:                 {report.test_rows:>12,}")
    typer.echo(f"future vintage test rows:  {report.future_vintage_test_rows:>12,}")
    typer.echo(f"grape vocab size:          {report.grape_vocab_size:>12,}")
    typer.echo(f"harmonize vocab size:      {report.harmonize_vocab_size:>12,}")
    typer.echo(f"output columns:            {report.output_columns:>12,}")


def _print_rating_report(report: RatingReport) -> None:
    typer.echo(f"features: {report.n_features}   train cells: {report.n_train:,}")
    for m in report.eval_metrics:
        typer.echo(
            f"[{m.split}] per-rating (noise floor {m.noise_floor:.4f} — "
            "even a perfect model cannot beat it)"
        )
        typer.echo(f"[{m.split}]   {'model':18s} RMSE={m.rmse:.4f}  MAE={m.mae:.4f}")
        for b in report.baselines.get(m.split, []):
            typer.echo(f"[{m.split}]   {b.name:18s} RMSE={b.rmse:.4f}  MAE={b.mae:.4f}")
        typer.echo(f"[{m.split}] cell-level (headline: mean rating per wine/vintage/age)")
        typer.echo(f"[{m.split}]   {'model':18s} RMSE={m.cell_rmse:.4f}  MAE={m.cell_mae:.4f}")
        for b in report.cell_baselines.get(m.split, []):
            typer.echo(f"[{m.split}]   {b.name:18s} RMSE={b.rmse:.4f}  MAE={b.mae:.4f}")
    typer.echo("bundles: " + ", ".join(report.bundle_names))


def _print_profile_report(report: ProfileReport) -> None:
    typer.echo(f"features: {report.n_features}")
    for m in report.metrics:
        typer.echo(
            f"[{m.split}] {m.target:14s} accuracy={m.accuracy:.4f}  macro_f1={m.macro_f1:.4f}"
        )
    typer.echo("bundles: " + ", ".join(report.bundle_names))


def _print_harmonize_report(report: HarmonizeReport) -> None:
    typer.echo(f"features: {report.n_features}  labels: {report.n_labels}")
    for sm in report.metrics:
        typer.echo(f"[{sm.split}] mean_f1={sm.mean_f1:.4f}  hamming={sm.hamming:.4f}")
        ranked = sorted(sm.per_label_f1.items(), key=lambda kv: kv[1], reverse=True)
        for label, f1 in ranked[:5]:
            typer.echo(f"  {label:22s} F1={f1:.4f}")
    typer.echo("bundles: " + ", ".join(report.bundle_names))


@train_app.command("rating")
def train_rating_cmd(
    config: Path = typer.Option("configs/rating_v1.yaml", "--config", help="Experiment yaml."),
    force: bool = typer.Option(False, "--force", help="Retrain even if bundles exist."),
    sample_frac: float | None = typer.Option(
        None, "--sample-frac", help="Subsample wines for a fast smoke run (e.g. 0.01)."
    ),
    track: bool = typer.Option(True, "--track/--no-track", help="Log the run to MLflow."),
) -> None:
    """Train the CatBoost rating regressor + quantile heads, report vs baselines."""
    report = train_rating(
        config, force=force, sample_frac=sample_frac, track=track, notify_fn=typer.echo
    )
    _print_rating_report(report)


@train_app.command("profile")
def train_profile_cmd(
    config: Path = typer.Option("configs/profile_v1.yaml", "--config", help="Experiment yaml."),
    force: bool = typer.Option(False, "--force", help="Retrain even if bundles exist."),
    sample_frac: float | None = typer.Option(
        None, "--sample-frac", help="Subsample wines for a fast smoke run (e.g. 0.01)."
    ),
    track: bool = typer.Option(True, "--track/--no-track", help="Log the run to MLflow."),
) -> None:
    """Train the Body and Acidity classifiers, report accuracy + macro-F1."""
    report = train_profile(
        config, force=force, sample_frac=sample_frac, track=track, notify_fn=typer.echo
    )
    _print_profile_report(report)


@train_app.command("harmonize")
def train_harmonize_cmd(
    config: Path = typer.Option("configs/harmonize_v1.yaml", "--config", help="Experiment yaml."),
    force: bool = typer.Option(False, "--force", help="Retrain even if bundles exist."),
    sample_frac: float | None = typer.Option(
        None, "--sample-frac", help="Subsample wines for a fast smoke run (e.g. 0.01)."
    ),
    track: bool = typer.Option(True, "--track/--no-track", help="Log the run to MLflow."),
) -> None:
    """Train the multilabel food-pairing model, report per-label F1 + Hamming."""
    report = train_harmonize(
        config, force=force, sample_frac=sample_frac, track=track, notify_fn=typer.echo
    )
    _print_harmonize_report(report)


@train_app.command("all")
def train_all_cmd(
    sample_frac: float | None = typer.Option(
        None, "--sample-frac", help="Subsample wines for a fast smoke run (e.g. 0.01)."
    ),
    track: bool = typer.Option(True, "--track/--no-track", help="Log the runs to MLflow."),
    configs_dir: Path = typer.Option(
        "configs", "--configs-dir", help="Directory holding the *_v1.yaml configs."
    ),
) -> None:
    """Run rating, profile, and harmonize training in sequence (default configs)."""
    typer.echo("=== rating ===")
    _print_rating_report(
        train_rating(
            configs_dir / "rating_v1.yaml",
            sample_frac=sample_frac,
            track=track,
            notify_fn=typer.echo,
        )
    )
    typer.echo("=== profile ===")
    _print_profile_report(
        train_profile(
            configs_dir / "profile_v1.yaml",
            sample_frac=sample_frac,
            track=track,
            notify_fn=typer.echo,
        )
    )
    typer.echo("=== harmonize ===")
    _print_harmonize_report(
        train_harmonize(
            configs_dir / "harmonize_v1.yaml",
            sample_frac=sample_frac,
            track=track,
            notify_fn=typer.echo,
        )
    )


@eval_app.command("ablations")
def eval_ablations_cmd(
    config: Path = typer.Option("configs/rating_v1.yaml", "--config", help="Experiment yaml."),
    arm: list[str] | None = typer.Option(
        None,
        "--arm",
        help="Run only these arms (repeatable). Default: all, `full` first.",
    ),
    sample_frac: float | None = typer.Option(
        None, "--sample-frac", help="Subsample wines for a fast smoke run (e.g. 0.01)."
    ),
    track: bool = typer.Option(True, "--track/--no-track", help="Log each arm to MLflow."),
) -> None:
    """Retrain the rating RMSE head per feature-block ablation arm.

    Arms: full (reference), no_terroir, no_producer, no_age. Every arm trains
    with the same config, data, and early-stopping fold, so the cell-level
    RMSE deltas are attributable to the dropped block. Expect roughly one
    RMSE-head training per arm — plan for hours on the full variant.
    """
    report = run_ablations(
        config, arms=arm or None, sample_frac=sample_frac, track=track, notify_fn=typer.echo
    )
    full = next((r for r in report.results if r.arm == "full"), None)
    for r in report.results:
        for m in r.metrics:
            delta = ""
            if full is not None and r.arm != "full":
                ref = next((fm for fm in full.metrics if fm.split == m.split), None)
                if ref is not None:
                    delta = f"  delta_cell_rmse={m.cell_rmse - ref.cell_rmse:+.4f}"
            typer.echo(
                f"[{m.split}] {r.arm:12s} cell_rmse={m.cell_rmse:.4f}  "
                f"cell_mae={m.cell_mae:.4f}  rmse={m.rmse:.4f}{delta}"
            )
    typer.echo(f"grid: {report.parquet_path}")


@eval_app.command("shap")
def eval_shap_cmd(
    split: str = typer.Option("test", "--split", help="test | future_vintage_test"),
    sample_cells: int | None = typer.Option(
        None,
        "--sample-cells",
        help="Cells sampled for the SHAP computation (default from config).",
    ),
) -> None:
    """Mean-|SHAP| ranking + terroir dependence plots for the rating model."""
    report = run_shap_analysis(
        split=split,  # type: ignore[arg-type]
        sample_cells=sample_cells,
        notify_fn=typer.echo,
    )
    typer.echo(f"top features by mean |SHAP| ({report.n_cells:,} {report.split} cells):")
    for name, value in report.top_features:
        typer.echo(f"  {name:28s} {value:.4f}")
    typer.echo("block totals (sum of mean |SHAP| per feature block):")
    for block, total in report.block_totals.items():
        typer.echo(f"  {block:10s} {total:.4f}")
    typer.echo(f"importance parquet: {report.importance_path}")
    for fig in report.figure_paths:
        typer.echo(f"figure: {fig}")


@eval_app.command("sanity")
def eval_sanity_cmd(
    wines: list[str] = typer.Argument(
        ..., help="Wine or winery name substrings (quote each, e.g. \"Sassicaia\")."
    ),
    top_pairings: int = typer.Option(5, "--top-pairings", help="Pairings shown per wine."),
) -> None:
    """Predict rating/body/acidity/pairings for named wines, next to the labels.

    A wine whose rows live in the train split shows a fitted value, not a
    forecast — the split column says which reading applies.
    """
    report = sanity_check(wines, top_pairings=top_pairings, notify_fn=typer.echo)
    for r in report.rows:
        typer.echo("")
        typer.echo(f"{r.winery_name} — {r.wine_name} ({r.region_name}, {r.vintage_year})")
        typer.echo(
            f"  split={r.split}  age_at_review={r.age_at_review}  n_ratings={r.n_ratings}"
        )
        typer.echo(
            f"  rating:  observed {r.observed_mean_rating:.2f}  ->  "
            f"predicted {r.predicted_rating:.2f}"
        )
        typer.echo(f"  body:    actual {r.actual_body or '?':18s} predicted {r.predicted_body}")
        typer.echo(
            f"  acidity: actual {r.actual_acidity or '?':18s} predicted {r.predicted_acidity}"
        )
        typer.echo(f"  pairings predicted: {', '.join(r.predicted_pairings)}")
        typer.echo(f"  pairings actual:    {', '.join(r.actual_pairings) or '(none)'}")
    if report.unmatched:
        typer.echo("")
        typer.echo("unmatched: " + "; ".join(report.unmatched))


if __name__ == "__main__":
    app()
