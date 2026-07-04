# 🍷 Vininator 3000

> Drink now or cellar? Wine rankings from CatBoost models trained on 21M ratings, NASA satellite weather, and soil chemistry per vintage.

The goal is pretty simple: It's to find the best wines. Vininator trains rating models on the [X-Wines](https://github.com/rogerioxavier/X-Wines) dataset (100k wines, 21M ratings) and turns them into four rankings:

- **Drink-now**: Which wines are predicted to be drinking best this year
- **Age-well**: Which wines' predicted trajectory still rises: the ones worth cellaring
- **Standouts per year**: For each drinking year 2026–2031, the bottles to open that year
- **Overperformers**: Wines predicted to punch far above their region/grape/vintage peer group

Under the hood: every rating in X-Wines is timestamped against its vintage, so `age_at_review` is a real model feature — sweeping it forward projects a wine's predicted rating to any future opening year. The feature set assumes terroir matters and encodes it properly: growing-season weather from [NASA POWER](https://power.larc.nasa.gov/) (MERRA-2 + CERES SYN1DEG) per `(region, vintage)` and [SoilGrids](https://soilgrids.org/) soil composition per region, next to grape, producer, and region. Supporting models predict body/acidity and food pairings to enrich the ranking tables. All rankings are published in [RESULTS.md](./RESULTS.md).


---

## Status

In development, currently Phase 5. See [PROJECT.md](./PROJECT.md) for the full plan and current phase.

This is a batch / CLI project — no hosted UI, no live API. The deliverable is the four ranking tables in [RESULTS.md](./RESULTS.md), backed by the trained models and the honest model-quality numbers that say how much to trust each list.

---

## Quickstart

### Prerequisites

- Python 3.12+
- [`uv`](https://github.com/astral-sh/uv) for dependency management

Weather data is pulled from the [NASA POWER Daily API](https://power.larc.nasa.gov/docs/services/api/temporal/daily/), which is free, public-domain, and needs no account or API key.

### Setup

```bash
# Clone and install Python deps
git clone https://github.com/vininator-3000/vininator-3000.git vininator
cd vininator
uv sync --extra notebook

# Configure
cp .env.example .env
# Edit .env if you want a non-default data dir or X-Wines variant

# Download X-Wines (test variant = 100 wines / 1k ratings, auto-fetched from GitHub)
uv run vininator data download

# Run the EDA notebook
uv run jupyter lab notebooks/01_eda.ipynb
```

To use a larger variant, download from the [X-Wines Google Drive](https://github.com/rogerioxavier/X-Wines#-availability):

```bash
# Drop XWines_Slim_1K_wines.csv and XWines_Slim_150K_ratings.csv into data/raw/
echo "VININATOR_XWINES_VARIANT=slim" >> .env
uv run vininator data download
```

### Running the full pipeline

```bash
# 1. Geocode regions (cached; first run ~30min for the full variant)
uv run vininator features geocode

# 2. Pull NASA POWER weather (cached, resumable; first run ~25min for the full variant)
uv run vininator features climate

# 3. Pull SoilGrids soil & terrain (cached, resumable; fast)
uv run vininator features soil

# 4. Assemble the training table
uv run vininator features build

# 5. Train the rating, profile, and Harmonize models
uv run vininator train rating --config configs/rating_v1.yaml
uv run vininator train profile --config configs/profile_v1.yaml
uv run vininator train harmonize --config configs/harmonize_v1.yaml
# Or you can run all of the above using
# uv run vininator train all

# 6. Measure each feature block's contribution by retraining with it dropped
uv run vininator eval ablations

# 7. Drink-now and age-well rankings (Phase 6)
uv run vininator recommend drink-now --opening-year 2026 --grape pinot-noir --max-vintage-age 5
uv run vininator recommend age-well  --opening-year 2026 --horizon 10 --grape nebbiolo
uv run vininator recommend standout-years --from-year 2026 --to-year 2031
uv run vininator recommend outliers --opening-year 2026

# 8. Regenerate RESULTS.md + reports/figures from the trained artifacts
uv run python scripts/build_results.py
```

---

## Project layout

```text
src/vininator/
  data/         X-Wines loader, geocoding (cached)
  features/     Climate (NASA POWER), soil & terrain (SoilGrids + DEM), terroir joiner, Harmonize parsing, feature assembly
  models/       CatBoost rating regressor (+ quantile heads), body / acidity classifiers, Harmonize multi-label
  eval/         Metrics, ablations, SHAP
  recommend/    the four rankings (drink-now, age-well, standout-years, outliers) by sweeping age_at_review
  cli.py        Typer CLI entrypoint

scripts/         build_results.py — regenerates RESULTS.md + reports/figures from artifacts
data/            raw → interim → processed (never edited after write)
notebooks/       01_eda, 02_climate, 03_soil, 04_rating, 05_harmonize, 06_ablations, 07_recommender, 08_results
configs/         One YAML per experiment
reports/figures/ SHAP plots, ablation charts, recommender summaries (embedded in RESULTS.md)
tests/           pytest suite
```

Full structure and rationale: [PROJECT.md](./PROJECT.md). Working conventions: [CLAUDE.md](./CLAUDE.md).

---

## Data

**Primary dataset:** [X-Wines](https://github.com/rogerioxavier/X-Wines) (Xavier 2023, MDPI BDCC). The full variant covers 100,646 wines and 21,013,536 ratings from 2012–2021, spanning 62 wine-producing countries.

Three variants are supported via the `VININATOR_XWINES_VARIANT` env var:

| variant | wines | ratings | source |
| --- | ---: | ---: | --- |
| `test` | 100 | 1,000 | GitHub raw (auto-fetched) |
| `slim` | 1,007 | 150,000 | Google Drive (manual drop) |
| `full` | 100,646 | 21,013,536 | Google Drive (manual drop) |

Each rating ships with `Date` (ISO timestamp) plus the rated `Vintage`, so the loader derives a per-row `age_at_review = year(Date) - Vintage` during normalization.

**License:** CC0 1.0 — public domain dedication. No usage restrictions.

**Weather (Phase 2):** Daily climate from the [NASA POWER Daily API](https://power.larc.nasa.gov/docs/services/api/temporal/daily/) (`~0.5° / ~55 km`), serving MERRA-2 temperature/precipitation and CERES SYN1DEG solar radiation. Free, public-domain, no registration. One JSON per region cached under `data/raw/nasa_power/`.

**Soil & terrain (Phase 2):** [SoilGrids](https://soilgrids.org/) (ISRIC) for topsoil composition — calcium carbonate (kalkgehalte), pH, texture, organic carbon, CEC, bulk density — plus SRTM elevation and derived slope. Pulled per region centroid, no auth required.

---

## Key design choices

- **Split by `WineID`, not by rating.** Same wine in train and test is leakage.
- **Future-vintage holdout** (train ≤ 2018, test 2019–2021) tests whether the model genuinely learned vintage-specific signal vs. memorized region averages.
- **`age_at_review` is the recommender's lever.** A real per-row feature (derived from `Date` and `Vintage`); sweeping it against the trained rating model projects predicted ratings forward to any opening year, which is how all four rankings are produced.
- **Producer (`WineryID`) aggregates computed on training fold only.** Standard target-leakage prevention.
- **CatBoost over manual encoding.** High-cardinality categoricals (`WineryID`, `RegionName`) handled natively.
- **Training happens on aggregated feature cells, not raw rating rows.** Rows sharing `(wine, vintage, age)` have identical features, so collapsing them is loss-exact for RMSE while shrinking the full variant ~7× — the difference between fitting in 16 GB of RAM and OOM.

---

## Evaluation

The rankings are only as good as the models behind them, so model quality is measured and published alongside every list:

- **Rating:** RMSE + MAE at two levels — per-rating (against leakage-safe baselines and the ~0.64 noise floor of user disagreement) and per wine-vintage cell (the headline: predicted vs. observed mean rating, where ranking quality actually lives). Quantile heads provide the confidence bands the rankings use to drop shaky picks.
- **Profile:** per-attribute accuracy and macro-F1 against the X-Wines Body / Acidity labels, per held-out wine-vintage.
- **Harmonize food pairings:** per-label F1 + Hamming loss on held-out wine-vintages.
- **Ablations (diagnostic):** drop terroir, drop producer, drop `age_at_review` — quantify what each block actually contributes. Terroir is an assumption baked into the feature set; the ablation reports honestly how much it earns its place.
- **SHAP** on the rating model to understand what's actually doing the work.

All metrics, ablation tables, SHAP plots, and the four ranking tables are published in [RESULTS.md](./RESULTS.md).

---

## Disclaimer

This is a personal hobby / learning research project. The list below captures the scoping decisions made to keep it tractable on one machine — each one is a trade-off a serious viticulture study would need to revisit:

- **Weather is 5 daily variables.** NASA POWER serves dozens of meteorological parameters; we pull daily min/mean/max temperature, precipitation total, and shortwave radiation total. No wind, humidity, evapotranspiration, soil moisture, dew point, etc. Enough for the canonical viticulture features (GDD, heat-spike days, growing-season precip, diurnal range, frost days) but loses signal that might matter for finer-grained predictions.
- **One grid cell per region.** NASA POWER returns the nearest 0.5° (~55 km) cell to the region's centroid. Large appellations (Bordeaux, Napa) collapse to a single point that may not represent the regional average growing conditions.
- **Soil is topsoil only (0–30 cm).** SoilGrids has deeper layers; we average its three topsoil bands (0–5 cm, 5–15 cm, 15–30 cm) and ignore everything below. Vine roots reach much deeper, but topsoil correlates with the variation we care about.
- **Reviews from X-Wines are subjective.** Each review carries equal weight regardless of who wrote it — no distinction between an amateur and a sommelier. A serious study would weight reviews by reviewer track record or expertise.
- **Geocode `result_type` blacklist.** Nominatim returns wildly inconsistent OSM entity types for wine regions — appellations come back tagged as `restaurant`, `volcano`, `peak`, `river`, etc. A whitelist would drop hundreds of real wine regions. We blacklist only obvious junk (`bus_stop`, `bank`, `school`, `fuel`, ...) and accept that a handful of real regions get caught: Yakima Valley tagged `college`, Patagonia tagged `atm`, Serra Gaúcha tagged `fuel`. ~3 % of the 1,422 geocoded regions drop out as collateral.
- **Some city centroids slip through.** Region strings like "Buenos Aires, Argentina" resolve to the city center via Nominatim, not the wine-growing area outside. The coordinates pass `status='ok'` but the soil pull then returns null over the urban grid. Downstream CatBoost handles the nulls; the rows aren't dropped.
- **One point per region, not per vineyard.** A wine from "Bordeaux" gets the climate + soil of the Bordeaux centroid even if it actually came from a south-facing slope in Saint-Émilion. A serious terroir model would work at parcel level.
- **`calcareous` is approximated via pH.** SoilGrids doesn't expose CaCO₃ directly. We flag soils as calcareous when `ph_h2o ≥ 7.5`, which catches the high-pH soils that limestone bedrock produces (Champagne, Chablis, Jerez) but isn't a literal carbonate measurement.
- **Drainage class is a coarse 4-bucket label.** Derived from clay% / sand% / calcareous via fixed thresholds: `chalky` > `clayey` (≥40 % clay) > `sandy` (≥60 % sand) > `loamy`. Real viticulture distinguishes far more soil regimes.
- **Slope without aspect.** Horn's algorithm on a 3×3 elevation grid gives us slope angle, but we ignore aspect (south-facing vs. north-facing matters a lot for sun exposure in NH vineyards).
- **`vintage_year` is treated as harvest year.** True for most New World wine, broadly true for Old World still wines, but conflates harvest-year and labelled-year for the rare wines (Champagne tirages, vintage Port) where they differ.
- **Climatology window is 1991–2018**, not the WMO-standard 1991–2020. Deliberate: ending the baseline at the training cutoff means the anomaly columns carry zero information leakage into the 2019–2021 future-vintage holdout.

---

## Acknowledgements

- **X-Wines dataset** — Xavier, R. de A. M. *X-Wines: A Wine Dataset for Recommender Systems and Machine Learning.* Big Data and Cognitive Computing, MDPI, 2023. [Paper](https://www.mdpi.com/2504-2289/7/1/20) · [Repo](https://github.com/rogerioxavier/X-Wines)
- **Weather data** — [NASA Langley Research Center POWER Project](https://power.larc.nasa.gov/), funded through the NASA Earth Science Directorate Applied Science Program. Underlying sources: MERRA-2 (meteorology) and CERES SYN1DEG (solar radiation).
- **SoilGrids** — ISRIC — World Soil Information. Hengl et al., *SoilGrids 2.0: producing soil information for the globe with quantified spatial uncertainty.* SOIL, 2021. [soilgrids.org](https://soilgrids.org/)

---

## License

Code: MIT (see `LICENSE`).
Data: X-Wines is CC0 1.0 (public domain).
