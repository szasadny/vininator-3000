"""Regenerate RESULTS.md from the trained bundles and recommendation parquets.

Phase 7 deliverable. Reads the model metadata, recomputes the numbers that were
not persisted (baseline grid, quantile coverage, per-class/per-label F1), re-runs
the four recommenders with default filters plus per-grape slices, and writes the
whole document — no hand-typed numbers. The interpretive notes are drawn from
PROJECT.md's established findings, so the output needs no manual editing.

Run it after the models and Phase 5 eval artifacts exist:

    PYTHONIOENCODING=utf-8 uv run python scripts/build_results.py
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from vininator.config import (
    MLFLOW_EXPERIMENT,
    MODEL_SEED,
    OUTLIER_MIN_PEER_WINES,
    RATING_BUNDLE,
    RECOMMEND_HORIZON_YEARS,
    RECOMMEND_TOP_N,
    RESULTS_MAJOR_GRAPES,
    RESULTS_OUTLIER_TABLE_N,
    RESULTS_SANITY_QUERIES,
    RESULTS_SHOWCASE_REGIONS,
    RESULTS_TABLE_TOP_N,
    STANDOUT_TOP_N,
    STANDOUT_YEAR_RANGE,
    WINE_SPLIT_SEED,
    get_settings,
)
from vininator.eval.report_data import (
    ablation_table,
    baseline_grid,
    harmonize_eval,
    load_test_wine_vintages,
    markdown_table,
    profile_eval,
    quantile_coverage,
    readme_disclaimer,
    shap_table,
    split_summary,
)
from vininator.eval.sanity import SanityRow, sanity_check
from vininator.models.artifacts import load_bundle
from vininator.models.tracking import git_sha
from vininator.recommend.age_well import recommend_age_well
from vininator.recommend.drink_now import (
    Bundles,
    RecommendFilters,
    build_candidates,
    load_recommend_bundles,
    recommend_drink_now,
)
from vininator.recommend.outliers import recommend_outliers
from vininator.recommend.standout_years import recommend_standout_years

_FIGURES = (
    ("SHAP summary — top features by mean |SHAP|", "shap_summary_test.png"),
    ("SHAP dependence — growing-degree days", "shap_dependence_gdd_10c_test.png"),
    ("SHAP dependence — GDD anomaly vs. climatology", "shap_dependence_gdd_10c_anom_test.png"),
    ("SHAP dependence — harvest-month precipitation", "shap_dependence_precip_harvest_mm_test.png"),
    ("SHAP dependence — calcareous soil flag", "shap_dependence_calcareous_test.png"),
)


def _note(message: str) -> None:
    print(message, flush=True)


def _grape_display(slug: str) -> str:
    """`"cabernet-sauvignon"` → `"Cabernet Sauvignon"` for a section heading."""
    return slug.replace("-", " ").replace("_", " ").title()


def _slug_fname(slug: str) -> str:
    """Filesystem-safe parquet stem for a slug (`"syrah/shiraz"` → `"syrah-shiraz"`)."""
    return slug.replace("/", "-").replace(" ", "-")


# ---------------------------------------------------------------------------
# Table display helpers (numbers pre-formatted; markdown_table renders the rest)
# ---------------------------------------------------------------------------


def _band(table: pl.DataFrame) -> list[str]:
    return [
        f"{p:.2f} ({lo:.2f}-{hi:.2f})"
        for p, lo, hi in zip(
            table.get_column("predicted_rating").to_list(),
            table.get_column("predicted_rating_lo").to_list(),
            table.get_column("predicted_rating_hi").to_list(),
            strict=True,
        )
    ]


def _drink_now_display(table: pl.DataFrame) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "Winery": table.get_column("winery_name").to_list(),
            "Wine": table.get_column("wine_name").to_list(),
            "Region": table.get_column("region_name").to_list(),
            "Vintage": table.get_column("vintage_year").to_list(),
            "Predicted (lo-hi)": _band(table),
            "Body": table.get_column("predicted_body").to_list(),
            "Acidity": table.get_column("predicted_acidity").to_list(),
            "Pairings": [", ".join(p) for p in table.get_column("top_pairings").to_list()],
        }
    )


def _age_well_display(table: pl.DataFrame) -> pl.DataFrame:
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


def _outlier_display(table: pl.DataFrame) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "Winery": table.get_column("winery_name").to_list(),
            "Wine": table.get_column("wine_name").to_list(),
            "Region": table.get_column("region_name").to_list(),
            "Vintage": table.get_column("vintage_year").to_list(),
            "Predicted": [f"{x:.2f}" for x in table.get_column("predicted_rating").to_list()],
            "Baseline": [f"{x:.2f}" for x in table.get_column("peer_baseline").to_list()],
            "Overperf": [f"{x:+.2f}" for x in table.get_column("overperformance").to_list()],
            "lo-hi": [
                f"{lo:.2f}-{hi:.2f}"
                for lo, hi in zip(
                    table.get_column("predicted_rating_lo").to_list(),
                    table.get_column("predicted_rating_hi").to_list(),
                    strict=True,
                )
            ],
        }
    )


def _sanity_display(rows: list[SanityRow]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "Wine": [f"{r.winery_name} - {r.wine_name}" for r in rows],
            "Region": [r.region_name for r in rows],
            "Vintage": [r.vintage_year for r in rows],
            "Split": [r.split for r in rows],
            "Observed": [f"{r.observed_mean_rating:.2f}" for r in rows],
            "Predicted": [f"{r.predicted_rating:.2f}" for r in rows],
            "Body (act/pred)": [f"{r.actual_body or '?'} / {r.predicted_body}" for r in rows],
            "Acidity (act/pred)": [
                f"{r.actual_acidity or '?'} / {r.predicted_acidity}" for r in rows
            ],
            "Pairings (pred)": [", ".join(r.predicted_pairings) for r in rows],
        }
    )


def _grape_drink_now(slug: str, cap: int | None, candidates: pl.DataFrame, bundles: Bundles):
    """Per-grape drink-now ranking, or `None` when nothing matched the filter."""
    settings = get_settings()
    try:
        return recommend_drink_now(
            filters=RecommendFilters(grape=slug, max_vintage_age=cap),
            top=RESULTS_TABLE_TOP_N,
            out_path=settings.report_tables_dir / f"drink_now_{_slug_fname(slug)}.parquet",
            candidates=candidates,
            bundles=bundles,
        )
    except ValueError:
        return None


def _grape_age_well(slug: str, candidates: pl.DataFrame, bundles: Bundles):
    """Per-grape age-well ranking, or `None` when nothing matched the filter."""
    settings = get_settings()
    try:
        return recommend_age_well(
            filters=RecommendFilters(grape=slug),
            top=RESULTS_TABLE_TOP_N,
            out_path=settings.report_tables_dir / f"age_well_{_slug_fname(slug)}.parquet",
            summary_path=settings.report_tables_dir
            / f"age_well_summary_{_slug_fname(slug)}.parquet",
            candidates=candidates,
            bundles=bundles,
        )
    except ValueError:
        return None


def _showcase_block(slug: str, display: str, candidates: pl.DataFrame, bundles: Bundles) -> str:
    """Region-filtered, non-monogrape drink-now table for a classic blend (§3.3)."""
    settings = get_settings()
    fname = _slug_fname(slug)
    cmd = f'vininator recommend drink-now --region "{slug}" --no-monogrape --top {RESULTS_TABLE_TOP_N}'
    try:
        report = recommend_drink_now(
            filters=RecommendFilters(region=slug, monogrape=False),
            top=RESULTS_TABLE_TOP_N,
            out_path=settings.report_tables_dir / f"drink_now_region_{fname}.parquet",
            candidates=candidates,
            bundles=bundles,
        )
    except ValueError:
        return f"#### {display}\n\nNo wines matched.\n"
    return (
        f"#### {display}\n\n```bash\n{cmd}\n```\n\n"
        + markdown_table(_drink_now_display(report.table))
        + "\n"
    )


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------


def _header(rating_meta: dict) -> str:
    return (
        "# Vininator 3000 - Results\n\n"
        "> Generated by `scripts/build_results.py` from the trained model bundles and "
        "recommendation parquets - do not hand-edit; edit the generator. Models trained at "
        f"git `{rating_meta['git_sha'][:7]}` (dataset `{rating_meta['dataset_hash']}`); "
        f"regenerated at git `{git_sha()[:7]}`."
    )


def _section_1(drink_now, age_well, outliers, model_cell, best_name, best_cell) -> str:
    parts = [
        "## 1. Headline: the picks",
        "",
        "**Top drink-now bottles (2026, all monogrape wines):**",
        markdown_table(_drink_now_display(drink_now.table.head(3))),
        "",
        "**Top cellar candidates (rising or late-peaking within 10 years):**",
        markdown_table(_age_well_display(age_well.table.head(3))),
        "",
        "**Biggest overperformers vs. peer baseline:**",
        markdown_table(_outlier_display(outliers.table.head(3))),
        "",
        (
            f"On held-out wines the rating model reaches a cell-level RMSE of {model_cell:.4f}, "
            f"against {best_cell:.4f} for the best leakage-safe baseline ({best_name}) - so the "
            "rankings clear the best peer average, which is the bar that makes them picks rather "
            "than noise."
        ),
    ]
    return "\n".join(parts)


def _section_2(rating_meta: dict, splits: pl.DataFrame) -> str:
    settings = get_settings()
    parts = [
        "## 2. Setup",
        "",
        f"- **Dataset variant:** {settings.xwines_variant}",
        f"- **Wine-split seed:** {WINE_SPLIT_SEED}   **model seed:** {MODEL_SEED}",
        f"- **Rating config:** {rating_meta['config_name']}",
        f"- **Git SHA (models):** {rating_meta['git_sha']}",
        f"- **Dataset hash:** {rating_meta['dataset_hash']}",
        (
            f"- **Experiment tracking:** MLflow local file store (`mlruns/`, experiment "
            f"`{MLFLOW_EXPERIMENT}`); runs are identified by git SHA + dataset hash - a file "
            "store has no shareable run links."
        ),
        "",
        "**Split sizes:**",
        markdown_table(splits),
    ]
    return "\n".join(parts)


def _grape_block(slug: str, cap: int | None, drink_now, age_well) -> tuple[str, str]:
    """Return the (drink-now, age-well) markdown blocks for one grape."""
    name = _grape_display(slug)
    cap_flag = f" --max-vintage-age {cap}" if cap is not None else ""
    dn_cmd = f"vininator recommend drink-now --grape {slug}{cap_flag} --top {RESULTS_TABLE_TOP_N}"
    aw_cmd = f"vininator recommend age-well --grape {slug} --top {RESULTS_TABLE_TOP_N}"

    if drink_now is None:
        dn = f"#### {name}\n\nNo monogrape {name} wines matched.\n"
    else:
        dn = (
            f"#### {name}\n\n```bash\n{dn_cmd}\n```\n\n"
            + markdown_table(_drink_now_display(drink_now.table))
            + "\n"
        )

    if age_well is None or age_well.table.is_empty():
        aw = f"#### {name}\n\nNo cellar candidates (every trajectory declines).\n"
    else:
        aw = (
            f"#### {name}\n\n```bash\n{aw_cmd}\n```\n\n"
            + markdown_table(_age_well_display(age_well.table))
            + "\n"
        )
    return dn, aw


def _section_3(grape_blocks: list[tuple[str, str]], showcase_blocks: list[str]) -> str:
    drink_now_blocks = "\n".join(b[0] for b in grape_blocks)
    age_well_blocks = "\n".join(b[1] for b in grape_blocks)
    return "\n".join(
        [
            "## 3. Drink-now and age-well rankings",
            "",
            "### 3.1 Drink-now (opening year 2026)",
            "",
            "Top monogrape wines predicted to drink best in 2026, per grape. Aromatic whites "
            "carry a 5-year freshness cap; cellar-style reds do not.",
            "",
            drink_now_blocks,
            "### 3.2 Age-well (2026 to 2036)",
            "",
            "Top monogrape wines whose predicted trajectory still rises or peaks late within the "
            "10-year horizon. `Clipped` marks bottles whose swept age left the trained range.",
            "",
            age_well_blocks,
            "### 3.3 Blends the monogrape filter hides",
            "",
            "The rankings default to single-varietal wines, so classic blends never reach 3.1-3.2. "
            "These region-filtered tables (`--no-monogrape`) surface them - Amarone della "
            "Valpolicella is a Corvina-based blend.",
            "",
            *showcase_blocks,
        ]
    )


def _section_4(standout, years: list[int], outliers) -> str:
    parts = [
        "## 4. Standouts and overperformers",
        "",
        "### 4.1 Standout wines of the year (2026 to 2031)",
        "",
        "One shortlist per drinking year - the wines predicted to be at their best that year. "
        "Produced by `vininator recommend standout-years`. A wine can recur across years when its "
        "trajectory plateaus.",
        "",
    ]
    for year in years:
        block = standout.table.filter(pl.col("opening_year") == year)
        parts.append(f"**{year}**")
        parts.append(markdown_table(_drink_now_display(block)))
        parts.append("")

    parts += [
        "### 4.2 Overperformer outliers (more special than expected)",
        "",
        "Wines whose predicted rating clears their leakage-safe peer baseline (per-(grape, region) "
        "and per-(region, vintage) train-fold means) by the largest margin, with the lower "
        f"confidence bound still above the baseline. Peer groups need at least {OUTLIER_MIN_PEER_WINES} "
        "distinct training wines to count. Produced by `vininator recommend outliers`.",
        "",
        markdown_table(_outlier_display(outliers.table)),
        "",
        "### 4.3 Ranking caveats",
        "",
        "- Rankings are conditional on wines *in X-Wines*, not the whole wine world.",
        "- Producer effects dominate, so lists skew toward well-rated wineries - signal, not bug.",
        "- Projections past ~10 years post-vintage are extrapolation; the trained age range is 0-71 "
        "years, and clipped rows are flagged.",
        "- Climate is region-centroid, not vineyard-parcel (see the README disclaimer).",
        "- Outliers are only as trustworthy as their baseline; sparse peer cells are excluded by the "
        f"{OUTLIER_MIN_PEER_WINES}-wine support gate.",
    ]
    return "\n".join(parts)


def _section_5(grids: dict[str, pl.DataFrame], coverage: pl.DataFrame) -> str:
    return "\n".join(
        [
            "## 5. Rating model quality",
            "",
            "Two levels per split: per-rating RMSE/MAE (comparable to the baseline literature but "
            "floored by the within-cell spread of user opinions, the `noise_floor` row) and "
            "cell-level weighted RMSE/MAE (predicted vs. observed mean rating per wine/vintage/age - "
            "the number that measures ranking quality).",
            "",
            "### 5.1 Held-out wines (random WineID split)",
            "",
            markdown_table(grids["test"]),
            "",
            "### 5.2 Future-vintage holdout (train <= 2018, test 2019-2021)",
            "",
            markdown_table(grids["future_vintage_test"]),
            "",
            "This split holds wines seen in training - only the vintage is new - so its RMSE is "
            "expected to be lower than 5.1's and the two are not comparable. 5.1 asks whether the "
            "model generalizes to a new wine; this asks whether it generalizes to a new year of a "
            "known wine.",
            "",
            "### 5.3 Confidence intervals",
            "",
            markdown_table(coverage),
            "",
            "The 0.1/0.9 quantile heads target a nominal 80% band on the wine-vintage mean rating; "
            "`coverage` is the ratings-weighted share of held-out cell means that fall inside it. "
            "These bands gate the overperformer table (4.2).",
        ]
    )


def _section_6(body, acidity, harm) -> str:
    return "\n".join(
        [
            "## 6. Profile + Harmonize models",
            "",
            "Quality of the enrichment columns (predicted body, acidity, pairings). Every metric "
            "counts each held-out wine-vintage once, matching how the models train.",
            "",
            "### 6.1 Body",
            "",
            markdown_table(body.headline),
            "",
            "Per-class F1 (test):",
            markdown_table(body.per_class_f1),
            "",
            "Confusion matrix (test, rows = actual, columns = predicted):",
            markdown_table(body.confusion),
            "",
            "### 6.2 Acidity",
            "",
            markdown_table(acidity.headline),
            "",
            "Per-class F1 (test):",
            markdown_table(acidity.per_class_f1),
            "",
            "Confusion matrix (test, rows = actual, columns = predicted):",
            markdown_table(acidity.confusion),
            "",
            "Acidity is ~79% High, so macro-F1 is the headline; accuracy alone rewards always "
            "predicting the majority class. The model trains with balanced class weights.",
            "",
            "### 6.3 Harmonize food-pairings",
            "",
            markdown_table(harm.headline),
            "",
            "Per-label F1 (test):",
            markdown_table(harm.per_label_f1),
            "",
            "Example wines (predicted vs. actual pairings):",
            markdown_table(harm.examples),
        ]
    )


def _section_7(ablations: pl.DataFrame, shap_top: pl.DataFrame, shap_block: pl.DataFrame) -> str:
    settings = get_settings()
    figs = []
    for caption, filename in _FIGURES:
        rel = (settings.figures_dir / filename).relative_to(settings.figures_dir.parents[1])
        figs.append(f"![{caption}]({rel.as_posix()})")
    return "\n".join(
        [
            "## 7. Diagnostics: SHAP + ablations",
            "",
            "### 7.1 SHAP analysis",
            "",
            "Top features by mean |SHAP| on the rating model (test cells):",
            markdown_table(shap_top),
            "",
            "Per-block rollup:",
            markdown_table(shap_block),
            "",
            *figs,
            "",
            "### 7.2 Ablations",
            "",
            "Cell-level RMSE per split when a feature block is dropped and the RMSE head retrained. "
            "`test_delta` / `fv_delta` are relative to the full model.",
            "",
            markdown_table(ablations),
            "",
            "- Terroir contributes nothing measurable: the deltas are noise on both splits, "
            "including the future-vintage split built to reveal vintage-weather learning. The "
            "region categorical plus vintage year already absorb region-keyed climate and soil at "
            "~55 km granularity.",
            "- `age_at_review` is the largest single contributor on held-out wines, validating the "
            "recommender lever. Its future-vintage delta is an artifact: 2019-2021 vintages rated by "
            "2021 span only 0-2 years of age.",
            "- Producer aggregates earn a real but modest gain on top of the raw WineryID categorical.",
        ]
    )


def _section_8(rows: list[SanityRow], unmatched: list[str]) -> str:
    parts = [
        "## 8. Qualitative sanity check",
        "",
        "Wines picked by hand and scored by the models. Queries are name substrings; a wine whose "
        "rows live in the train split shows a fitted value, not a forecast (see `Split`). "
        '"Beaujolais" is read as its Gamay grape.',
        "",
        markdown_table(_sanity_display(rows)) if rows else "_No sanity wines matched._",
    ]
    if unmatched:
        parts += ["", "Unmatched queries: " + ", ".join(unmatched) + "."]
    return "\n".join(parts)


def _section_9(disclaimer: str) -> str:
    return "\n".join(
        [
            "## 9. Limitations & caveats",
            "",
            "Scoping decisions carried over from the README:",
            "",
            disclaimer,
            "",
            "Model-specific caveats surfaced during evaluation:",
            "",
            "- Quantile bands can cross (`lo > hi`) on a few cells; they are reported as-is rather "
            "than clamped, and coverage is quantified in 5.3.",
            "- X-Wines mislabels a handful of non-wines (e.g. a Nebbiolo-tagged grappa), which the "
            "grape filter can surface - the recommender scores whatever the dataset labels.",
            "- Coverage is skewed toward popular regions; picks in obscure regions rest on thinner "
            "support.",
        ]
    )


def _section_10() -> str:
    commands = "\n".join(
        [
            "uv run vininator data download",
            "uv run vininator features geocode",
            "uv run vininator features climate",
            "uv run vininator features soil",
            "uv run vininator features terroir",
            "uv run vininator features build",
            "uv run vininator train all",
            "uv run vininator eval ablations",
            "uv run vininator eval shap",
            "uv run vininator recommend drink-now --opening-year 2026",
            "uv run vininator recommend age-well --opening-year 2026 --horizon 10",
            "uv run vininator recommend standout-years --from-year 2026 --to-year 2031",
            "uv run vininator recommend outliers --opening-year 2026",
            "uv run python scripts/build_results.py",
        ]
    )
    return "\n".join(
        [
            "## 10. Reproduction",
            "",
            "From a fresh clone with the X-Wines full variant in `data/raw/` and `.env` configured:",
            "",
            f"```bash\n{commands}\n```",
        ]
    )


def _acknowledgements() -> str:
    return "\n".join(
        [
            "## Acknowledgements",
            "",
            "- **X-Wines dataset** - Xavier 2023, MDPI BDCC. CC0 1.0.",
            "- **NASA POWER** - LaRC POWER Project. Underlying: MERRA-2 + CERES SYN1DEG.",
            "- **SoilGrids** - ISRIC. Hengl et al., 2021.",
        ]
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def main() -> None:
    settings = get_settings()
    settings.report_tables_dir.mkdir(parents=True, exist_ok=True)
    rating_meta = load_bundle(RATING_BUNDLE).meta

    # Order matters for memory: the baseline grid holds the (projected) train
    # frame, so it runs and frees before the recommender builds candidates.
    _note("[1/6] baseline grid")
    grids = baseline_grid()
    _note("[2/6] quantile coverage")
    coverage = quantile_coverage()

    _note("[3/6] profile + harmonize eval")
    wv = load_test_wine_vintages()
    body = profile_eval("body_label", wv=wv)
    acidity = profile_eval("acidity_label", wv=wv)
    harm = harmonize_eval(wv=wv)
    del wv

    splits = split_summary()

    _note("[4/6] recommenders (canonical + per grape)")
    bundles = load_recommend_bundles()
    candidates = build_candidates(_note)

    drink_now = recommend_drink_now(
        top=RECOMMEND_TOP_N, candidates=candidates, bundles=bundles, notify_fn=_note
    )
    age_well = recommend_age_well(
        top=RECOMMEND_TOP_N,
        horizon=RECOMMEND_HORIZON_YEARS,
        candidates=candidates,
        bundles=bundles,
        notify_fn=_note,
    )
    standout = recommend_standout_years(
        top=STANDOUT_TOP_N, candidates=candidates, bundles=bundles, notify_fn=_note
    )
    outliers = recommend_outliers(
        top=RESULTS_OUTLIER_TABLE_N, candidates=candidates, bundles=bundles, notify_fn=_note
    )

    grape_blocks: list[tuple[str, str]] = []
    for slug, cap in RESULTS_MAJOR_GRAPES:
        _note(f"    grape: {slug}")
        dn = _grape_drink_now(slug, cap, candidates, bundles)
        aw = _grape_age_well(slug, candidates, bundles)
        grape_blocks.append(_grape_block(slug, cap, dn, aw))

    showcase_blocks = [
        _showcase_block(slug, display, candidates, bundles)
        for slug, display in RESULTS_SHOWCASE_REGIONS
    ]
    del candidates, bundles

    _note("[5/6] sanity check")
    queries = [q for q, _ in RESULTS_SANITY_QUERIES]
    preferred = {q: v for q, v in RESULTS_SANITY_QUERIES if v is not None}
    sanity = sanity_check(queries, preferred_vintages=preferred, max_matches=1, notify_fn=_note)

    _note("[6/6] diagnostics + render")
    ablations = ablation_table()
    shap_top, shap_block = shap_table()
    disclaimer = readme_disclaimer()

    best = _best_baseline(grids["test"])
    model_cell = rating_meta["metrics"]["test"]["cell_rmse"]

    years = list(range(STANDOUT_YEAR_RANGE[0], STANDOUT_YEAR_RANGE[1] + 1))
    sections = [
        _header(rating_meta),
        _section_1(drink_now, age_well, outliers, model_cell, best[0], best[1]),
        _section_2(rating_meta, splits),
        _section_3(grape_blocks, showcase_blocks),
        _section_4(standout, standout.years_covered or years, outliers),
        _section_5(grids, coverage),
        _section_6(body, acidity, harm),
        _section_7(ablations, shap_top, shap_block),
        _section_8(sanity.rows, sanity.unmatched),
        _section_9(disclaimer),
        _section_10(),
        _acknowledgements(),
    ]
    document = "\n\n---\n\n".join(sections) + "\n"
    _write_atomic(settings.results_md, document)
    _note(f"wrote {settings.results_md}")


def _best_baseline(test_grid: pl.DataFrame) -> tuple[str, float]:
    """The lowest cell-RMSE leakage-safe baseline (name, rmse) on the test split."""
    baselines = test_grid.filter(
        ~pl.col("predictor").is_in(["model (CatBoost)", "noise_floor"])
    ).sort("cell_rmse")
    return baselines.get_column("predictor")[0], float(baselines.get_column("cell_rmse")[0])


def _write_atomic(target: Path, content: str) -> None:
    """Write text via a tmp sibling then rename; LF newlines even on Windows."""
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8", newline="\n")
    tmp.replace(target)


if __name__ == "__main__":
    main()
