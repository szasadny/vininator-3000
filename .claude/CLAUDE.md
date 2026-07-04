# Vininator 3000 — Wine Rating & Profile Predictor

## Domain

A wine recommender: the goal is to find the best wines in the X-Wines dataset — drink-now, age-well, standout-per-year, and overperformer-outlier rankings, produced by sweeping `age_at_review` against a trained CatBoost rating model. The machinery: rating regression (+ quantile bands) over grape, region, vintage, producer, and `age_at_review`, augmented with a **terroir feature block** (NASA POWER daily climate per `(region, vintage_year)` + SoilGrids soil/terrain per `region`); supporting body/acidity and food-pairing models enrich the ranking tables. Terroir mattering is a working assumption of the feature design, not the research question — its actual contribution is audited via ablation and reported honestly.

Primary dataset: **X-Wines** (`rogerioxavier/X-Wines` on GitHub, CC0 1.0). Full variant: ~100k wines / 21M ratings, every rating timestamped with its rated vintage — which is what makes `age_at_review` a real per-row feature. No review text, no images: all targets are structured labels.

The deliverable is trained models plus findings published in RESULTS.md — batch / CLI only, no hosted API or frontend.

For the full plan, phases, and sequencing, see [PROJECT.md](../PROJECT.md). That document is the source of truth for what we're building and in what order. This file is the source of truth for *how* we work.

---

## Stack

| Layer | Technology |
| --- | --- |
| Language | Python 3.12 |
| Env / deps | `uv` (lockfile committed) |
| Data | `polars` (preferred over pandas for the main tables) |
| ML | `catboost` (primary), `scikit-learn` (utilities) |
| Weather | NASA POWER Daily API (MERRA-2 + CERES SYN1DEG, JSON, no auth) |
| Soil | SoilGrids REST API (ISRIC), no auth |
| Terrain | SRTM 30 m via Open-Elevation |
| Geocoding | `geopy` (Nominatim) |
| Experiment tracking | `mlflow` (local file store in `mlruns/`) |
| CLI | `typer` + `rich` |
| Lint / test | ruff + pytest |

---

## Project Structure

```text
src/vininator/
  data/         # X-Wines loader, geocoding (cached, resumable)
  features/     # climate.py (NASA POWER → GDD/precip/anomalies), soil.py, terroir.py (joiner), text.py (Harmonize parsing), build.py (assemble final table)
  models/       # dataset.py (shared feature contract + cell aggregation), rating.py, profile.py, harmonize.py, artifacts.py, tracking.py
  eval/         # metrics, ablations, SHAP
  recommend/    # drink_now.py, age_well.py, standout_years.py, outliers.py (Phase 6)
  cli.py        # typer CLI entrypoint: `vininator train rating`, etc.

data/
  raw/          # X-Wines CSVs + parquets, NASA POWER JSON pulls — never modified after write
  interim/      # geocoded regions, climate.parquet, soil.parquet, terroir.parquet
  processed/    # final feature parquets (train/test/future_vintage) + recommendation parquets
  models/       # trained bundles (.cbm + .meta.json), gitignored

notebooks/      # exploration only — see PROJECT.md §5 for the numbered list
configs/        # yaml per experiment
scripts/        # build_results.py — emits RESULTS.md tables + figures
tests/
```

**Navigation rule:** when working on a task, read only the folder relevant to that task. Grep before scanning. Notebooks are for exploration; production code lives in `src/vininator/`.

---

## Conventions

**Python**
- Ruff (formatter + linter). Type hints everywhere. `from __future__ import annotations` at the top of every module.
- All public functions get docstrings; explain *why*, not *what*.
- Pydantic v2 for settings (`config.py`). Plain frozen dataclasses for internal config and reports.
- Sync code throughout — this is a batch/CLI project; external fetches are rate-limited sequential loops, not async.
- Pathlib only — never `os.path.join`.
- No hardcoded paths. All paths come from `src/vininator/config.py` (which reads env vars with defaults).

**General**
- `.env` for local config; never committed.
- No commented-out code, no dead code, no `TODO`-as-placeholder. If it's not done, leave a real comment explaining what and why.

---

## ML & Data Standards

These are the rules that protect the *headline result*. They are non-negotiable.

- **Split by `wine_id`, not by review.** Same wine in train and test is leakage. Every split function must enforce this — including the early-stopping validation fold inside the trainers.
- **Future-vintage holdout.** In addition to the random wine-id split, hold out vintages 2019–2021 as a separate test set. Note it contains *wines seen in training* (only the vintage is new), so its RMSE is not comparable to the wine-split RMSE — it answers "does the model generalize to a new year of a known wine", not "to a new wine".
- **Train on aggregated feature cells, never raw rating rows.** All features are wine-level, `(region, vintage)`-level, or the age itself, so rating rows sharing `(wine_id, vintage_year, age_at_review)` are duplicates. `models/dataset.aggregate_rating_cells` collapses them (loss-exact for weighted RMSE, ~7× smaller); wine-level targets (body/acidity/pair_*) go through `aggregate_wine_vintage` (~21× smaller, drops `age_at_review`). This is what makes full-variant training fit in 16 GB RAM. Any new feature that varies inside a cell must extend the cell keys or become metadata — there's a test guarding this.
- **Report the rating headline at cell level.** Per-rating RMSE is floored by within-cell user disagreement (~0.64 vs. a global std of ~0.74 on the full variant), so terroir deltas drown in it. The headline metric is weighted cell-level RMSE (predicted vs. observed mean rating per wine/vintage/age); per-rating RMSE is reported next to the logged `noise_floor` for baseline comparability.
- **No target leakage in producer aggregates.** Producer mean-rating / std / n_reviews features are computed **on the training fold only**, then applied to test. Never compute on the full dataset.
- **Cache every external call.** NASA POWER, SoilGrids, and Nominatim are all rate-limited and intermittently fail. Every external fetch goes through a function that checks a parquet/sqlite/json cache first, writes the result atomically, and is resumable across restarts.
- **Raw data is immutable.** Files in `data/raw/` are never modified after write. Cleaning and joining happen on the way to `data/interim/` and `data/processed/`.
- **Track every experiment.** MLflow/W&B from run #1. Hyperparameters, dataset hash, git SHA, metrics, feature list — all logged. "I'll start tracking once it works" never happens.
- **Report ablations honestly.** Terroir is an assumption in the feature design, not the thesis — but its contribution is still measured (rating-with-terroir vs. without, cell-level). If terroir adds 1% RMSE, that's the number that goes in RESULTS.md — don't bury it, don't inflate it.
- **Sample weighting.** Use `log(1 + n_ratings)` per wine (summed per cell after aggregation). A wine with 5000 ratings is a different signal than a wine with 5.
- **The recommender never re-engineers features.** Phase 6 scores only wines already in X-Wines by sweeping `age_at_review`; vintage (and therefore terroir) is held constant, so there is no live terroir fetch and no inference path to the upstream APIs.

---

## External Solutions First

Before implementing something in-house, check whether a stable, maintained library already solves it.

- **Boosted trees** → `catboost`. Native categoricals (don't manually target-encode).
- **Weather data** → NASA POWER Daily API. MERRA-2 quality via clean JSON, no auth, public domain. Don't scrape weather sites; don't reach for raw NetCDF / xarray when a clean JSON wrapper exists.
- **Geocoding** → `geopy` with Nominatim. Don't write a CSV of regions by hand.
- **Experiment tracking** → MLflow (already wired in `models/tracking.py`). Don't roll your own logging.
- **General rule:** if a maintained PyPI package solves ≥80% of the problem, use it. Reinventing is more bugs and more maintenance.

---

## Maintainability

Write code for the developer maintaining it 12 months from now.

- **No magic values.** Thresholds, paths, hyperparameter defaults, growing-season month ranges, vocabulary sizes — all live in `src/vininator/config.py` or a yaml in `configs/`.
- **Single source per piece of behaviour.** Splitting logic, feature assembly, model loading — each defined once and reused. If you find yourself writing the same block in a second file, lift it.
- **Layering.**
  - `data/` → reads raw, returns dataframes. No feature engineering.
  - `features/` → takes dataframes, returns dataframes with new columns. No model training.
  - `models/` → takes processed dataframes, returns trained model artifacts + metrics. No file I/O outside the canonical paths.
  - `recommend/` → loads saved bundles, scores wines at opening years. Never re-trains, never re-engineers features inline.
  - `cli.py` → the only place that orchestrates phases end-to-end.
- **Notebooks are not production.** Exploration lives in `notebooks/`. Once a finding is real, the code moves into `src/vininator/`. Notebooks may import from the package but the package never imports from a notebook.
- **Think at scale — the box has 16 GB RAM and no CUDA GPU.** The full X-Wines ratings table is 21M rows; use polars lazy + `scan_parquet` + predicate pushdown, aggregate before the polars→pandas boundary, and free large frames (`del` + `gc.collect()`) as soon as their derived artifacts exist. The raw frame does not fit next to its pandas copy.
- **Reproducibility.** Set seeds. Log dataset hashes. The train script should produce the same metrics on a fresh checkout given the same config.
- **Explicit over clever.** A longer, obvious implementation beats a one-liner that requires context.
- **No half-finished features.** Leave code in the last working state. No disabled blocks, no broken branches in main.

---

## Working Approach

**Before writing:**

- Read PROJECT.md if you don't remember the phase you're in or what comes next.
- Read only the files you'll touch, plus their direct imports.
- Grep for the existing pattern before writing new code — match it exactly.
- If a stable external library solves the problem, prefer it.
- **Ask when genuinely split.** If two architecturally sound options exist with real trade-offs (e.g., "store the cache as parquet or sqlite"), present them and ask. Don't pick arbitrarily.
- **Ask before assuming on external resources.** If the task needs a HuggingFace token or a downloaded artifact that isn't in the repo, stop and ask — don't write code that silently fails on a missing credential.

**While writing:**

- Scope changes tightly — a bug fix changes the bug, a feature adds the feature.
- Check for existing abstractions before building new ones.
- Flag observed debt in your response; don't silently fix it.
- ML rules never relax for "just to see if it works." No leakage shortcuts, no commented-out splits.

**After writing:**

- Run `ruff check` and `pytest` before declaring done.
- Verify importing modules still resolve.
- **Turn manual checks into tests.** If you verified something by running a script and eyeballing output, capture it as a pytest test — the specific case plus a general test of the surrounding behaviour.

**Maintaining this file:**

- After adding or removing a top-level folder, update the Project Structure section in the same change.
- When a cross-cutting rule changes (new ML standard, new layering boundary), update this file as part of the same change.
- For complex situational context spanning multiple prompts, create `.claude/<topic>.md` and add one reference line here — delete it when no longer relevant.
- Never add changelogs or task notes here; git tracks what changed.
