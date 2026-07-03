# Vininator 3000 — Project Plan

A wine rating + tasting-note predictor trained on the X-Wines dataset, with a terroir feature pipeline combining NASA POWER daily climate (MERRA-2 + CERES SYN1DEG) with SoilGrids soil and terrain properties.

---

## 1. Goals

Three prediction targets, all structured (X-Wines ships no review text, so generative tasting-note descriptors are off the table):

1. **Rating** — regression on the 1–5 X-Wines score. The headline target.
2. **Structured profile** — body and acidity (X-Wines ships these as labels) + alcohol % (`ABV`).
3. **Food-pairing profile** — multi-label classifier over the top-N Harmonize pairings (e.g. "beef", "shellfish", "blue cheese"). This is the closest proxy we have to a tasting profile without review text.

**Inputs:** grape variety/blend, region (hierarchical), per-rating vintage year, producer (`WineryID`), `age_at_review`, plus a derived **terroir feature block** that combines `region × vintage` weather from NASA POWER (MERRA-2 + CERES SYN1DEG) with `region`-level soil and terrain properties from SoilGrids.

**Core objective:** build a predictive model that estimates a wine's sensory profile — rating, structure, and pairing profile — from structured wine metadata, the bottle age at the moment of opening, and vintage-specific terroir conditions. The model is then applied to two downstream questions:

- **Drink-now ranking** — for a target opening year (e.g. 2026), which monogrape wines in the dataset are predicted to be drinking best *right now*? Optional filters by grape and by maximum vintage age (e.g. "sub-5-years" for fresh-style wines).
- **Age-well ranking** — for the same set of wines, project predicted ratings forward by sweeping `age_at_review` over future opening years. Wines whose predicted trajectory still rises (or peaks late) are the ones to cellar.
- **Standout-of-the-year ranking** — for each of the next five drinking years (2026, 2027, 2028, 2029, 2030, 2031), surface the handful of wines predicted to drink best *in that specific year*. This is the drink-now score evaluated at six successive opening years and curated into a per-year "what to open this year" shortlist.
- **Overperformer outliers** — wines predicted to be more special than their peers would suggest: the predicted rating sits well above the leakage-safe peer-group baseline (per-`(GrapeMajority, RegionName)` and per-`(RegionName, Vintage)` mean) by a margin that survives the prediction-interval band. These are the "punching above their weight" picks the model flags as quietly exceptional.

All four rankings are produced offline and published in [RESULTS.md](./RESULTS.md) alongside the modeling findings.

---

## 2. Dataset — X-Wines

- **Source:** [`rogerioxavier/X-Wines`](https://github.com/rogerioxavier/X-Wines) on GitHub. The `test` variant ships in-repo (auto-fetched by the loader); `slim` and `full` are on the X-Wines Google Drive (manual download).
- **License:** **CC0 1.0** (public domain). No restrictions.
- **Variants:**
  - `test` — 100 wines / 1,000 ratings (smoke-test only).
  - `slim` — 1,007 wines / 150,000 ratings (good for iteration without paying the full-data cost).
  - `full` — **100,646 wines / 21,013,536 ratings**, 1,056,079 users, 30,510 wineries, 2,160 regions, 62 countries, collected 2012–2021.
- **Wines schema:** `WineID, WineName, Type, Elaborate, Grapes (list[str]), Harmonize (list[str], food pairings), ABV, Body, Acidity, Code, Country, RegionID, RegionName, WineryID, WineryName, Website, Vintages (list[int])`.
- **Ratings schema:** `RatingID, UserID, WineID, Vintage (int), Rating (1–5, half-steps), Date (ISO timestamp)`. The loader derives `age_at_review = year(Date) - Vintage` during normalization.
- **Why this dataset:** every rating ships with an exact timestamp and the rated `Vintage`, so `age_at_review` is a real per-row feature (the headline reason we dropped WineSensed, which has no review timestamps). Full structured attributes are on every wine, not just ~5%.
- **No images** are included — this dataset is metadata-only.

---

## 3. Architecture overview

```
X-Wines CSVs (test | slim | full)
        │
        ▼
  data/raw/xwines_wines.parquet
  data/raw/xwines_ratings.parquet   (with derived `age_at_review`)
        │
        ├─► geocoded regions ──┬─► NASA POWER weather pull ──► climate.parquet
        │                       │                                │
        │                       └─► SoilGrids pull   ──► soil.parquet
        │                                                        │
        │                                          ──► terroir.parquet (climate ⨝ soil)
        │                                                        │
        ▼                                                        ▼
  feature assembly (joins wine + terroir + parsed Harmonize features)
        │
        ▼
  data/processed/train.parquet
        │
        ├─► CatBoost rating regressor
        ├─► CatBoost body / acidity classifiers
        └─► CatBoost Harmonize multi-label classifier
        │
        ▼
  Recommender (CLI) — sweep `age_at_review` over opening years
        │
        ├─► data/processed/recommendations_drink_now.parquet
        ├─► data/processed/recommendations_age_well.parquet
        ├─► data/processed/recommendations_standout_years.parquet  (2026→2031)
        ├─► data/processed/recommendations_outliers.parquet        (overperformers)
        │
        ▼
  RESULTS.md  (curated headline tables, ablations, SHAP, recommender outputs)
```

The deliverable is the trained models plus the findings published in RESULTS.md. No hosted product, no live API — everything is batch / CLI. The recommender operates only on wines already present in X-Wines (no live terroir fetch for unseen vintages), varying `age_at_review` against the model to project drink-now and age-well rankings.

---

## 4. Phases

### Phase 1 — Data acquisition & EDA

- Pull X-Wines via `vininator data download` (test variant from GitHub; slim/full are dropped into `data/raw/` after a Google Drive download).
- Loader normalizes both CSVs into parquets, parses `Grapes` and `Vintages` list columns, and derives `age_at_review = year(Date) - Vintage`.
- EDA notebook: schema + missingness on both tables, `age_at_review` distribution (sanity-check negative ages and outliers), rating distribution, ratings-per-year coverage, grape / region / country / winery cardinality, ratings-per-wine distribution, `(RegionName, Vintage)` cells with `>= 5` wines.

**Deliverable:** `notebooks/01_eda.ipynb` + a written summary of what's actually usable.

### Phase 2 — Terroir feature pipeline

**Region → coordinates**
- Geocode each unique region string to lat/lon. Use Nominatim (OSM) or GeoNames. Rate-limit politely (1 req/sec for Nominatim).
- Cache aggressively — geocode each region once, ever. A few thousand unique regions, all cached to `data/interim/geocode.parquet`.
- Imprecise names ("Bordeaux") → use appellation centroid. Accept the lossiness.
- The same `(lat, lon)` feeds both the climate and the soil pipeline below.
- **Audit `result_type` before downstream use.** Nominatim returns the OSM entity type ("administrative", "region", "city", "village", "house", "monument", …). Some X-Wines region strings resolve to the *wrong* entity — e.g. "Buenos Aires, Argentina" → city centre, "Scanderbeg, Albania" → a public square named after the historical figure. These rows pass `status='ok'` but their coordinates don't land in vineyard country, so SoilGrids returns null and the terroir signal is junk. Before any model trains on these features, filter the geocode parquet to keep only `result_type ∈ {administrative, region, county, state, locality, suburb, isolated_dwelling, hamlet, village}` (or a similar whitelist refined against the actual distribution). The soil pull is unaffected — CatBoost handles the nulls — but the filter must happen before training so the bad rows don't enter the training set.

**Weather (NASA POWER Daily API)**

**Strategy: one HTTP request per region.** [NASA POWER](https://power.larc.nasa.gov/) is a free, public-domain REST/JSON endpoint serving daily MERRA-2 meteorology and CERES SYN1DEG solar radiation at ~0.5° / ~55 km resolution, going back to 1981. A single request fetches all of `CLIMATE_YEAR_RANGE` × all five daily variables for one (lat, lon). After the PR2.5 geocode blacklist there are **~1,377 requests total**, cached as one JSON per region under `data/raw/nasa_power/{slug}.json`. No account, no token, no NetCDF parsing.

**Why NASA POWER over Open-Meteo / ERA5-Land?** We initially pivoted *to* Open-Meteo (an ERA5-Land wrapper, 0.1° / ~11 km) because it avoided Copernicus CDS chunking, then pivoted *off* it when the free tier turned out to bill per data point — 30 years × 5 variables × one coordinate exceeded the daily quota in a single region pull. NASA POWER's coarser resolution is the cost of staying free; for the growing-degree-day / heat-spike / harvest-precip aggregates we actually compute, the ~5× resolution downgrade is negligible compared to the centroid-vs-vineyard error already baked into region geocoding. POWER's temperature data is itself derived from MERRA-2 (sibling to ERA5-Land in lineage), so the underlying science isn't a step down — just the grid.

**Variables pulled per region** (NASA POWER variable IDs → our feature math, units verified via live API metadata):

| NASA POWER variable | Units | Used for |
| --- | --- | --- |
| `T2M_MIN` | °C | Spring frost days, diurnal range |
| `T2M` | °C | GDD, climatology baseline, anomalies |
| `T2M_MAX` | °C | Heat-spike days, diurnal range |
| `PRECTOTCORR` | mm/day | Growing-season + harvest-month precip |
| `ALLSKY_SFC_SW_DWN` | MJ/m²/day | Sunshine / cloud-cover proxy |

Units land in exactly the form `compute_climate_features` expects — no unit conversion in the loader. POWER returns `-999.0` for days where the source product was unavailable; `load_nasa_power_daily` coerces that sentinel to null so downstream null-detection (`is_partial`, climatology skip) works the same as it did with Open-Meteo's explicit nulls.

**Growing season mask** (applied at feature time, not at fetch time, so we can recompute thresholds without re-fetching):

- Northern hemisphere (lat ≥ 0): **April–October** of vintage_year.
- Southern hemisphere (lat < 0): **October of (vintage_year − 1) – April** of vintage_year.

**Features computed per `(region, vintage)`:**

- **Growing Degree Days (GDD)** — `Σ max(Tmean_daily − 10°C, 0)` across the season.
- **Total growing-season precipitation** (mm).
- **Harvest-month precipitation** (last 30 days of the growing-season window).
- **Heat spike days** — count of days `Tmax > 35°C`.
- **Spring frost days** — count of April–May (or Oct–Nov SH) days `Tmin < 0°C`.
- **Mean diurnal temperature range** — average of `(Tmax − Tmin)` across the season.
- **Total growing-season solar radiation** (MJ/m²).
- **Anomaly vs. 1991–2018 regional climatology** for each of the above. The baseline window deliberately ends at the training cutoff (2018) rather than the WMO-standard 1991–2020, so the anomaly column carries **zero information leakage** into the 2019–2021 future-vintage holdout. If the climatology used 2019–2020 it would peek at the held-out years and overstate the model's out-of-sample skill on vintages we haven't trained on. The climatology is computed once per region from the same per-region pull (no extra HTTP calls). A hot year in Bordeaux means something different from a hot year in Mendoza, so the anomaly often predicts better than the absolute value.

**Partial-vintage handling.** A growing-season window may extend outside the pulled year range (e.g. a Southern-hemisphere 1991 vintage's October–December 1990 leg falls before 1991) or carry nulls inside the season (POWER's `-999.0` sentinel after parsing, for days where the source product was unavailable). The batch path computes features on whatever days are present and sets `is_partial=True` on the row — both for missing dates and for null `tmean` inside the season. Rows are kept; the downstream model can decide whether to use them. The Phase 7 `TerroirProvider` is the only path that *substitutes* climatology for the partial season — batch never does.

**NASA POWER compliance & politeness**

- **Public domain data.** POWER data are released without licence restrictions. Vininator embeds an acknowledgement of LaRC and the underlying MERRA-2 / CERES SYN1DEG sources in parquet metadata — required courtesy, not a legal blocker.
- **Polite request spacing.** POWER doesn't publish a hard rate limit but the docs warn that hammering the same coordinate triggers opaque blocking. 1 s between submissions keeps the 1,377-region pull at ~23 minutes wall time and well clear of "hammering" territory.
- **Backoff on transient failures.** Exponential backoff (2s, 8s, 30s, 60s, 120s, 300s) on network blips and 5xx. POWER returns no structured `Retry-After`, so every retry uses the same schedule. Persistent failure after all retries raises `NasaPowerError` and stops the run.
- **Resume from disk, not memory.** Each region's JSON is written atomically (`tmp` → `rename`). On restart, the loader skips any region whose `.json` already exists and is non-empty. No central state file — the cache *is* the state.
- **User-Agent** identifies the project: `vininator-3000/0.1`. POWER doesn't require a contactable operator, but identifying ourselves is good manners.
- **Attribution.** Every derived artifact (`climate.parquet`, trained models) gets the POWER attribution embedded in its metadata: *"Weather data from the NASA Langley Research Center POWER Project, funded through the NASA Earth Science Directorate Applied Science Program. Underlying sources: MERRA-2 (meteorology) and CERES SYN1DEG (solar radiation)."*

**Sizing (one-time, end-to-end):** 1,377 HTTP requests, each returning a multi-megabyte JSON spanning 31 years. Aggregated `climate.parquet` is small (one row per `(region, vintage_year)` × 31 years × ~1.4k regions = ~43k rows).

**Soil & terrain (SoilGrids 250m via ISRIC REST API)**

Soil composition is a defining component of terroir and is largely time-invariant — pulled per `region`, not per `(region, vintage)`. Free, no auth required, well-documented.

- For each unique region centroid, query SoilGrids for the topsoil (0–30 cm) profile. Use a small spatial neighbourhood (e.g. 1 km buffer, mean aggregation) so single-pixel noise is smoothed out.
- Extract:
  - **Calcium carbonate content (`CaCO3`, "kalkgehalte")** — the classic chalky-soil signal (Champagne, Chablis, Jerez).
  - **pH (in H2O)** — acidity of the soil itself; correlates with grape acid retention.
  - **Soil texture** — clay, sand, silt percentages. Together they determine drainage and heat retention.
  - **Soil organic carbon (`SOC`)** — proxy for fertility; vines on rich soils tend to over-crop and underperform.
  - **Cation exchange capacity (`CEC`)** — nutrient-holding capacity.
  - **Bulk density** — proxy for compaction and rooting depth.
- Derive a few interpretable composites:
  - **Drainage class** — coarse bucket from sand% and clay% (`sandy`, `loamy`, `clayey`, `chalky` when CaCO3 is dominant).
  - **Calcareous flag** — boolean, CaCO3 above a region-relative threshold.
- Pull **elevation and slope** from a DEM (SRTM 30m via `elevation` or Open-Elevation) at the same centroid. Slope and aspect influence sun exposure and frost drainage; elevation correlates with diurnal range.
- Cache raw SoilGrids responses per region to `data/interim/soil_raw/`. The API is friendly but flaky — resume must work.

**Deliverable:** `data/interim/climate.parquet` keyed by `(region, vintage_year)`, `data/interim/soil.parquet` keyed by `region`, and a joined `data/interim/terroir.parquet`. Source files: `src/vininator/features/climate.py` (NASA POWER), `src/vininator/features/soil.py`, `src/vininator/features/terroir.py` (joiner).

### Phase 3 — Feature engineering & dataset assembly

Build the final training table with these blocks:

**Wine identity**
- Grape variety (high cardinality; for blends, multi-hot encode top ~50 grapes + "other")
- Producer/winery (very high cardinality — let CatBoost handle natively, or target-encode)
- Country, region, sub-region, appellation (hierarchical categoricals)
- Wine type (red/white/rosé/sparkling/dessert/fortified)

**Wine context**
- Vintage year
- `age_at_review` (per-rating, derived from the rating timestamp — the lever the Phase 6 recommender sweeps)
- Alcohol % (`ABV`) — X-Wines ships no price, so there is no price feature

**Climate block** (from Phase 2) — six absolute metrics + six anomalies = ~12 numerical features, joined on `(region, vintage_year)`.

**Soil & terrain block** (from Phase 2) — CaCO3, pH, clay/sand/silt %, organic carbon, CEC, bulk density, elevation, slope + derived `drainage_class` (categorical) and `calcareous` (boolean), joined on `region` only. Treat as static per region.

**Producer aggregates** (carefully, to avoid leakage)
- Producer mean rating, std, n_reviews — computed **on the training fold only**.

*(No text-derived features: X-Wines ships no review text. Body and Acidity come as structured labels and are targets, not parsed features.)*

**Canonicalization phase**
- Canonicalize producer, grape, and region names to reduce duplication and improve grouping consistency.

**Sample weight:** `log(1 + n_ratings)` per wine (summed per cell after aggregation, see below).

**Cell aggregation (critical for both compute and metrics):**

The processed parquets stay one-row-per-rating, but no trainer fits on them directly — `models/dataset.py` collapses to feature cells at load time:

- **Rating model:** every feature is wine-level, `(region, vintage)`-level, or the age itself, so rows sharing `(wine_id, vintage_year, age_at_review)` carry identical feature vectors. Collapsing a cell to (mean rating, summed weight) is *gradient-identical* to raw rows for weighted RMSE — same trees, same early-stopping point — while the full variant shrinks ~7× (15.5M rows → 2.25M cells). This is what makes full-data training fit in 16 GB RAM; without it every full run OOM'd.
- **Profile + Harmonize models:** Body / Acidity / `pair_*` are constant per wine, so these trainers collapse to one row per `(wine_id, vintage_year)` (~21× smaller: 15.5M rows → 746k wine-vintages) and drop `age_at_review` — a wine-level label cannot depend on when reviewers happened to rate. Eval happens at the same level, so popular wines no longer dominate the metrics.
- **Quantile heads change meaning deliberately:** trained on cells they band the wine-vintage mean rating (model uncertainty about wine quality — what the recommender's overperformer filter needs) instead of the spread of individual user opinions (so wide the filter would never fire).

**Splitting strategy (critical):**

- Split by **`WineID`**, not by review. Same wine in train and test is leakage.
- Held-out test: 15% of wines.
- Additional "future vintage" test: train on vintages ≤ 2018, test on 2019–2021. Reveals whether the model learned terroir or just memorized region averages.
- Use grouped cross-validation by `WineID` during development and hyperparameter tuning.
- Optional: grouped validation by producer or region to test generalization beyond memorized winery/region effects.

**Deliverable:** `data/processed/train.parquet`, `data/processed/test.parquet`, `data/processed/future_vintage_test.parquet`.

### Phase 4 — Modeling

We predict the structured attributes the dataset actually ships. No retrieval, no similar-wine surfacing, no generative tasting notes — X-Wines has no review text, so flavor descriptors are out of scope and we don't try to fake it. What we *can* do is predict the labels that exist, well, and on wines and vintages the model has never seen.

**Rating model — CatBoost regression** *(primary target)*

Handles high-cardinality categoricals natively, trains on CPU, is the boring correct choice. Use `age_at_review` as a first-class numerical feature — this is what makes the Phase 6 recommender possible, since varying `age_at_review` at inference projects the predicted rating to any opening year. Trained on aggregated `(wine, vintage, age)` cells (see Phase 3) with the CPU speed knobs from `configs/rating_v1.yaml` (`max_ctr_complexity: 1`, `one_hot_max_size: 8`, `thread_count: 10`).

**Evaluation happens at two levels**, because per-rating RMSE has a hard floor: the within-cell spread of user opinions is ~0.64 on the full variant against a global std of ~0.74, so even a *perfect* model scores ~0.64 per-rating and the whole model-quality game plays out in a 0.10-RMSE window. The trainers log this floor as `noise_floor` per split.

1. **Per-rating RMSE/MAE** — comparable 1:1 to the baselines below; read it against `noise_floor`, never as an absolute score.
2. **Cell-level RMSE/MAE** *(headline)* — predicted vs. observed mean rating per `(wine, vintage, age)` cell, weighted by ratings per cell, with the same baseline grid evaluated at the same level. The user-opinion noise averages out here, so this is where the terroir ablation delta is actually visible.

**Baselines to beat** (measured on the full X-Wines variant, in-sample / optimistic; out-of-sample on a `WineID` split will be looser):

1. Global mean — **RMSE ~0.74** (= `std(Rating)`, since predicting the mean for everyone gives an RMSE equal to the rating standard deviation; the global mean itself is **3.89**, not 3.5).
2. Per-`WineID` mean — **RMSE ~0.65** in-sample. Unavailable at test time under a wine-level split — replace with per-`WineryID` mean (a leakage-safe proxy) for a held-out baseline.
3. Per-`(WineID, Vintage)` mean — **RMSE ~0.63** in-sample. Same wine-split caveat.
4. Per-`(RegionName, Vintage)` mean — leakage-safe under both wine and future-vintage splits. The number we actually need to beat in production.
5. Per-`(GrapeMajority, RegionName)` mean — same as (4) but stratified by the dominant grape.

The headline experiment is "rating with terroir block" vs. "rating without terroir block", anchored to baselines (1) and (4-5), reported at the cell level.

Additional modeling: quantile regression for prediction intervals / confidence bands. Trained on the same aggregated cells, the 0.1/0.9 heads band the *wine-vintage mean rating* — model uncertainty about wine quality, which is what the recommender needs to flag low-confidence picks (per-rating quantiles would band user disagreement instead, far too wide to be useful).

**Profile model — CatBoost multi-class** *(secondary targets)*

Classifiers trained against the X-Wines `Body` and `Acidity` labels (no `Tannin` ships in X-Wines — derive from review text in Phase 3 if we want it).

- `Body` — 5 ordinal classes: `{Very light-bodied, Light-bodied, Medium-bodied, Full-bodied, Very full-bodied}`. Heavy skew: 44% Full, 34% Medium, 11% Very Full, 10% Light, 1% Very Light.
- `Acidity` — 3 ordinal classes: `{Low, Medium, High}`. **Very heavy skew: 79% High, 18% Medium, 3% Low** — single-class baseline already gets ~79% accuracy; report macro-F1, not accuracy, and consider class weights.

Same feature set as rating minus `age_at_review` — both trainers run on the wine-vintage aggregated table (the labels are wine-level constants; see Phase 3), and eval counts each held-out wine-vintage once instead of once per rating.

**Harmonize food-pairing model — CatBoost multi-label** *(secondary target)*

Train a multi-label classifier to predict the top-N Harmonize food-pairing vector (top-30 pairings, already a Phase 3 feature) from the full feature set minus `age_at_review` (wine-vintage aggregated, like profile). Output is a probability per pairing label; threshold per-label using validation-fold F1.

This is the closest stand-in we have for a tasting profile without review text — "this wine pairs with grilled red meat and aged hard cheese" is a structural claim about body, tannin, and intensity, even if it isn't a flavor descriptor. Evaluated with per-label F1 and Hamming loss on held-out wines.

**Why no retrieval / similar-wine surfacing.** An earlier draft of this phase tried to surface "3–5 similar real wines" via k-NN over predicted Harmonize + profile + terroir vectors. We dropped it: the goal is to predict the *quality of future wines* (i.e. wines the model hasn't seen — including older vintages a user wants to drink today and current vintages projected to age), not to retrieve neighbours from the training set. Retrieval is also fundamentally a recommendation problem, not a prediction problem, and conflating the two muddies the headline result (rating-with-terroir vs. without). What we lose: a way to show "wines that taste like this." What we keep: clean predictions for rating, body, acidity, and pairings that the Phase 6 recommender can sweep over opening years.

### Phase 5 — Evaluation & analysis

- **Rating:** per-rating and cell-level RMSE + MAE on held-out wines and held-out future vintages, next to the logged `noise_floor`. The cell-level number is the headline. Interpretation note: the future-vintage split contains wines seen in training (only the vintage is new), so its RMSE is expected to be *lower* than the wine-split RMSE — the two splits answer different questions and are not comparable to each other.
- **Profile:** per-attribute accuracy and macro-F1 against the X-Wines Body / Acidity labels, one row per held-out wine-vintage.
- **Harmonize:** per-label F1 + Hamming loss on held-out wine-vintages' actual food-pairing vectors.
- **SHAP / feature importance on the rating model** — what actually matters? Particular focus on whether the terroir block contributes signal *above* producer + region + grape.
- **Ablations:** drop terroir, drop producer, drop `age_at_review`. Quantify each block's marginal contribution on the cell-level metric. Ablations retrain the **RMSE head only** — the quantile heads don't affect the ablation conclusion and would triple the cost.
- **Qualitative sanity check:** pick 10 wines I personally know, predict ratings + Body/Acidity + pairings, eyeball it. Disagreements get written up in RESULTS.md — they're more interesting than the agreements.

### Phase 6 — Drink-now & age-well recommender

Once the rating model is trained, `age_at_review` is the lever that turns it into a recommender. Every wine in X-Wines has a fixed `(grape, region, vintage_year, terroir_block, producer, ...)`. Sweeping `age_at_review` over future opening years projects the model's predicted rating trajectory for that wine — no live terroir fetch required, since vintage_year (and therefore the terroir features) is held constant.

#### Drink-now ranking

For a target opening year (default: current year), score every wine in the dataset at `age_at_review = opening_year - vintage_year`. Rank by predicted rating. Filters:

- `--grape <variety>` — restrict to wines whose `GrapeMajority` matches (canonicalized name).
- `--max-vintage-age <n>` — drop wines with `opening_year - vintage_year > n`. The headline use case is `--max-vintage-age 5` for fresh-style wines.
- `--monogrape` (default on) — restrict to single-varietal wines (one entry in `Grapes`). Blends are out of scope for the recommender because the "best Pinot Noir to drink now" question doesn't have a clean answer when half the bottle is something else.
- `--region <name>` — optional region filter.

Output: a parquet of `(WineID, WineryName, WineName, RegionName, Vintage, predicted_rating, predicted_rating_lo, predicted_rating_hi, predicted_body, predicted_acidity, top_pairings)` sorted by predicted rating descending. The `_lo` / `_hi` columns come from the Phase 4 quantile model and are used to drop low-confidence picks from the curated tables in RESULTS.md.

#### Age-well ranking

Same wine set, but score each wine at multiple future opening years (e.g. `opening_year ∈ {current, current+3, current+5, current+10}`) and surface wines whose predicted-rating trajectory:

- still rises across the window (cellar candidates), or
- peaks late (peak age > current year), or
- holds steady (long plateau).

Output: a parquet keyed by `(WineID, opening_year, predicted_rating, predicted_rating_lo, predicted_rating_hi)` long-format, plus a derived wide summary (`predicted_peak_year`, `predicted_peak_rating`, `slope_to_peak`) for ranking.

#### Standout-of-the-year ranking

The drink-now score, run across the **next five drinking years (2026 → 2031)**, then curated into one shortlist per year. For each opening year in the window, score every (monogrape, by default) wine at `age_at_review = opening_year - vintage_year` and take the top-N by predicted rating. The point is editorial: "if you're opening a bottle in 2028, here are the wines the model says are at their best *that* year." The same `--grape` / `--max-vintage-age` / `--region` filters as drink-now apply.

Implementation note: this is `drink_now` looped over the year window — no new model, no new feature math. It reuses the same scoring path so a wine's per-year rank is consistent with its standalone drink-now rank.

Output: a long-format parquet keyed by `(opening_year, rank, WineID, WineryName, WineName, RegionName, Vintage, predicted_rating, predicted_rating_lo, predicted_rating_hi, predicted_body, predicted_acidity, top_pairings)`, with `opening_year ∈ {2026..2031}` and `rank` the within-year position.

#### Overperformer outliers

Wines the model flags as **more special than expected** — predicted to outscore their peer group by a margin that isn't explained by region/grape/vintage averages alone. For each wine at its drink-now opening year, compute:

- `peer_baseline` — the leakage-safe expectation for that wine: the max (or mean, configurable) of the per-`(GrapeMajority, RegionName)` and per-`(RegionName, Vintage)` baseline means already built for Phase 4. These are computed **on the training fold only**, consistent with the no-leakage rule.
- `overperformance = predicted_rating - peer_baseline`.

Rank by `overperformance` descending, and keep only rows where the **lower** confidence bound still clears the baseline (`predicted_rating_lo > peer_baseline`) so a wide prediction interval can't manufacture a fake outlier. The result is a shortlist of quietly exceptional wines — not the highest absolute ratings (those skew to famous producers), but the biggest positive surprises relative to where the wine "should" land.

Output: a parquet keyed by `(WineID, WineryName, WineName, RegionName, Vintage, predicted_rating, peer_baseline, overperformance, predicted_rating_lo, predicted_rating_hi)` sorted by `overperformance` descending.

#### CLI

```bash
vininator recommend drink-now \
  --opening-year 2026 \
  --grape pinot-noir \
  --max-vintage-age 5 \
  --top 50 \
  --out data/processed/recommendations_drink_now.parquet

vininator recommend age-well \
  --opening-year 2026 \
  --horizon 10 \
  --grape nebbiolo \
  --top 50 \
  --out data/processed/recommendations_age_well.parquet

vininator recommend standout-years \
  --from-year 2026 \
  --to-year 2031 \
  --top 10 \
  --out data/processed/recommendations_standout_years.parquet

vininator recommend outliers \
  --opening-year 2026 \
  --top 50 \
  --out data/processed/recommendations_outliers.parquet
```

All subcommands print the top-N to stdout (rich table) and write a parquet for downstream RESULTS.md table generation. `standout-years` and `outliers` share the `--grape` / `--max-vintage-age` / `--region` / `--monogrape` filters with `drink-now`.

**Deliverable:** `src/vininator/recommend/{drink_now.py, age_well.py, standout_years.py, outliers.py}`, CLI wired into `cli.py`, and the four recommendation parquets under `data/processed/`. `standout_years.py` and `outliers.py` reuse `drink_now`'s scoring path rather than re-implementing it.

### Phase 7 — Findings publication (RESULTS.md)

The final deliverable. A self-contained writeup at the project root that someone can read top-to-bottom and understand both what we built and what we found.

#### Structure

1. **Headline result.** One paragraph: did terroir help, by how much (RMSE delta), and what's the honest takeaway. No burying.
2. **Setup.** Dataset variant used, train/test/future-vintage split sizes, seed, git SHA, MLflow / W&B run links.
3. **Rating model.** Per-rating and cell-level RMSE + MAE tables for the random wine-split and the future-vintage split, against the baseline grid and the noise floor. Confidence intervals. The ablation grid (drop terroir / drop producer / drop age) as a single table.
4. **Profile + Harmonize models.** Macro-F1 by class, confusion matrices for Body and Acidity, per-label F1 and example pairings for Harmonize.
5. **SHAP analysis.** Top features by mean absolute SHAP, plus 3–4 dependence plots for the most interesting terroir variables (GDD, harvest-month precip, calcareous flag, ...).
6. **Drink-now and age-well tables.** Curated headline tables from Phase 6: top-10 monogrape picks for 2026 drinking across the major grapes (Cabernet Sauvignon, Pinot Noir, Chardonnay, Riesling, Nebbiolo, Tempranillo, ...), and top-10 age-well picks per grape with their projected peak years. Each table cites the recommender command that produced it.
7. **Standout-of-the-year and overperformer tables.** One shortlist of standout wines per drinking year across 2026 → 2031 (the "what to open this year" picks), and the overperformer-outlier table — wines predicted to be more special than their peer-group baseline, ranked by `overperformance` with their confidence band. Each cites the recommender command that produced it.
8. **Qualitative sanity check.** The 10 known-wines results from Phase 5, with commentary on where the model agreed and where it didn't.
9. **Limitations & caveats.** Pulled from the disclaimer block already in the README, plus model-specific caveats discovered during evaluation.
10. **Reproduction.** The exact CLI sequence to rebuild every artifact from a fresh clone.

**Deliverable:** `RESULTS.md` at the repo root, plus the supporting figures under `reports/figures/` (SHAP plots, ablation charts, recommender summary tables). RESULTS.md is regenerated from the trained models and recommendation parquets — no manual numbers typed by hand. A small `scripts/build_results.py` (or notebook 08) emits the markdown tables and figure files.

---

## 5. Project structure

```
vininator/
├── pyproject.toml          # uv for deps; ruff + pytest configured
├── README.md
├── PROJECT.md              # this file
├── RESULTS.md              # Phase 7 deliverable — generated from trained artifacts
├── CLAUDE.md               # operating rules for Claude Code
├── data/
│   ├── raw/                # X-Wines CSVs + parquets, NASA POWER JSON pulls, SoilGrids responses
│   ├── interim/            # geocoded regions, climate.parquet, soil.parquet, terroir.parquet
│   └── processed/          # final feature parquets + recommendation parquets
├── src/vininator/
│   ├── data/
│   │   ├── load.py         # X-Wines loader
│   │   └── geocode.py      # region → lat/lon (cached)
│   ├── features/
│   │   ├── climate.py      # NASA POWER → GDD, precip, anomalies
│   │   ├── soil.py         # SoilGrids + DEM → CaCO3, pH, texture, slope, ...
│   │   ├── terroir.py      # join climate + soil into the terroir block
│   │   ├── text.py         # parse Harmonize food pairings → multi-hot features
│   │   └── build.py        # assemble final feature table
│   ├── models/
│   │   ├── rating.py       # CatBoost rating regressor (+ quantile heads)
│   │   ├── profile.py      # body/acidity classifiers
│   │   └── harmonize.py    # Harmonize multi-label classifier
│   ├── eval/
│   │   ├── metrics.py
│   │   └── ablations.py
│   ├── recommend/
│   │   ├── drink_now.py        # score wines at opening_year, rank by predicted rating
│   │   ├── age_well.py         # sweep opening_year forward, find rising/peaking trajectories
│   │   ├── standout_years.py   # per-year standout shortlists across 2026→2031
│   │   └── outliers.py         # overperformers: predicted rating vs. peer-group baseline
│   └── cli.py              # typer CLI: vininator train rating, vininator recommend, etc.
├── scripts/
│   └── build_results.py    # emits RESULTS.md tables + reports/figures from trained artifacts
├── notebooks/              # 01_eda, 02_climate, 03_soil, 04_rating, 05_harmonize, 06_ablations, 07_recommender, 08_results
├── configs/                # yaml per experiment (rating_v1.yaml, etc.)
├── reports/figures/        # SHAP plots, ablation charts, recommender summaries embedded in RESULTS.md
└── tests/
```

---

## 6. Stack

| Layer | Choice | Why |
| --- | --- | --- |
| Language | Python 3.12 | Standard for ML |
| Env / deps | `uv` | Fast, modern, lockfiles work |
| Data wrangling | `polars` | Faster than pandas at 800k rows; lazy is nice |
| Modeling | `catboost` | Native categorical support; no manual encoding |
| Text | `ast`, `re` | Harmonize parsing (no review text in X-Wines, so no embeddings) |
| Weather | NASA POWER Daily API | MERRA-2 + CERES SYN1DEG via clean JSON REST, no auth |
| Soil | SoilGrids REST API (ISRIC) | Free, no auth, global 250 m coverage |
| Terrain | SRTM 30 m via `elevation` or Open-Elevation | Free; elevation + slope per centroid |
| Geocoding | `geopy` (Nominatim) | Free; respect rate limits |
| Tracking | `mlflow` *or* `wandb` | Choose one; track from day one |
| CLI | `typer` + `rich` | Subcommands for data, features, train, recommend; rich tables in stdout |
| Reporting | Matplotlib + Markdown templating | RESULTS.md + reports/figures/, generated from artifacts |
| Lint/test | ruff + pytest | Standard |

---

## 7. Realistic things to know

- **NASA POWER pulls take ~23 minutes sequentially.** One HTTP request per region (~1,377 total after the geocode blacklist), polite 1s spacing, exponential backoff on 5xx, resume-from-disk per JSON file. The cache *is* the state — a half-finished run restarts cleanly. Attribution must be embedded in derived artifacts (LaRC POWER + MERRA-2 + CERES SYN1DEG lineage). Public domain, no auth, no quota. We pivoted off Open-Meteo's ERA5-Land wrapper after discovering its free tier bills per data point — 30 years × 5 vars × one coordinate exceeded the daily quota in a single region pull. POWER's ~55 km cells are coarser than ERA5-Land's ~11 km, but for growing-season aggregates the difference is negligible next to the centroid-vs-vineyard error.
- **Climatology window is 1991–2018, not the WMO-standard 1991–2020.** The baseline used to compute climate anomalies ends at the training cutoff so the anomaly column contains zero information about the 2019–2021 future-vintage holdout. Reporting that "the model generalizes to future vintages" would be a lie if the anomalies it trained on already peeked at those vintages.
- **SoilGrids is fast but flaky.** Single-pixel queries can be noisy and the endpoint occasionally 5xxs. Always buffer-and-average, always retry with backoff, always cache.
- **Geocoding has rate limits.** Nominatim asks for 1 req/sec. A few thousand regions is fine, just plan for it.
- **Geocode `result_type` lies sometimes.** Nominatim happily resolves a wine-region string to a city, monument, or random POI when the appellation isn't in OSM under that exact name. The resulting `status='ok'` row points at the wrong place, and SoilGrids returns null on top of that wrong location. Audit the `result_type` distribution after the geocode pull and filter to a known-good whitelist (administrative / region / locality / hamlet / etc.) before training. Examples encountered in PR2 smoke: "Buenos Aires" → `city`, "Scanderbeg" → `square`.
- **The full X-Wines variant is 21M ratings.** Plenty of data, but for iteration always work off the `slim` variant (150k ratings) or `--sample-frac` — full is for the final training run.
- **Full-data training only fits in 16 GB RAM because of cell aggregation.** The raw train split is 15.5M rows (~6 GB in polars, doubled at the pandas boundary, plus CatBoost's quantized pool) — every pre-aggregation full run died on memory. Aggregated (2.25M cells for rating, ~450k wine-vintages for profile/harmonize) the whole Phase 4 sequence runs in an evening on the 12-core box with `thread_count: 10` keeping the machine usable. Don't undo this by materializing raw-row pools "just to check something" — check on `slim`.
- **Per-rating RMSE is floored at ~0.64 by user disagreement.** A flat validation curve near that floor is the ceiling being hit, not a learning-rate problem. Diagnose model changes on the cell-level metric.
- **Producer (`WineryID`) will dominate everything.** Be ready for the result that terroir adds a few percent RMSE improvement on top of producer + region + grape. That's still a real, interesting result — just not the headline "weather predicts wine" story. Frame the project honestly around this from the start.
- **Coverage is skewed toward popular regions.** The model will be best at well-represented regions and worse at obscure ones. Check and report this explicitly.
- **Splits matter.** Split by `WineID`, not by rating. Future-vintage split reveals real terroir learning vs. memorization.
- **`age_at_review` is real but bounded.** `Date` covers 2012-2021, so ratings of pre-2012 vintages are over-represented at high ages; ratings of recent vintages are absent at high ages. Treat `age_at_review` as a feature, not a target.
- **License.** X-Wines is CC0 (public domain). No restrictions on intermediate artifacts or trained models.
