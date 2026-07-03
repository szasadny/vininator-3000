# Vininator 3000 — Results

> **Status:** placeholder. Phase 4 (modeling) is still in progress. This file is regenerated end-to-end by `scripts/build_results.py` once the trained models, ablation runs, and recommender outputs exist. The structure below is the contract — sections will be populated with real numbers, tables, and figures as each phase completes. Do not hand-edit; edit the generator.

---

## 1. Headline result

*To be filled by `scripts/build_results.py`.*

One paragraph: did the terroir block (NASA POWER climate + SoilGrids soil) improve rating prediction over a producer + region + grape + `age_at_review` baseline, by how much (cell-level RMSE delta on the future-vintage split), and what's the honest takeaway. Reported even if the answer is "no meaningful improvement" — see [PROJECT.md §7](./PROJECT.md#7-realistic-things-to-know) on framing.

---

## 2. Setup

- **Dataset variant:** *(test | slim | full — filled at generation)*
- **Train / test / future-vintage split sizes:** *(filled)*
- **Seed:** *(filled)*
- **Git SHA:** *(filled)*
- **Experiment tracking run:** *(MLflow / W&B link, filled)*
- **Reproduction command:** see [§9](#9-reproduction).

---

## 3. Rating model

All rating tables report **two levels**: per-rating RMSE/MAE (comparable to the PROJECT.md baseline numbers, floored by the within-cell spread of user opinions — the `noise_floor` row) and **cell-level** weighted RMSE/MAE (predicted vs. observed mean rating per `(wine, vintage, age)` cell — the headline, where terroir deltas are visible instead of drowned in user noise).

### 3.1 Held-out wines (random `WineID` split)

*Table: per-rating and cell-level RMSE + MAE for the trained model and the baseline grid — global mean, per-`WineryID` mean, per-`(RegionName, Vintage)` mean, per-`(GrapeMajority, RegionName)` mean — plus the `noise_floor` for the per-rating columns.*

### 3.2 Future-vintage holdout (train ≤ 2018, test 2019–2021)

*Same metrics, on the vintage-generalization split. Interpretation note (rendered with the table): this split contains wines seen in training — only the vintage is new — so its RMSE is expected to be lower than §3.1's and the two are **not** comparable to each other. §3.1 asks "does the model generalize to a new wine"; this section asks "does it generalize to a new year of a known wine" — the terroir question.*

### 3.3 Ablations

*Table with rows = ablated block (none / − terroir / − producer / − `age_at_review`) and columns = cell-level RMSE on each split, plus the delta against the full model. RMSE head only (quantile heads don't affect the conclusion).*

### 3.4 Confidence intervals

*Per-prediction lo / hi from the quantile heads — bands on the wine-vintage mean rating, not on individual user ratings — summarized as coverage of observed cell means on the held-out set.*

---

## 4. Profile + Harmonize models

All profile/harmonize metrics count each held-out **wine-vintage once** (not once per rating), matching how the models train — per-rating metrics would be dominated by popular wines.

### 4.1 Body

*Confusion matrix and per-class F1 against the 5 X-Wines Body classes. Macro-F1 reported headline, not accuracy — the class skew (44% Full-bodied) makes accuracy uninformative.*

### 4.2 Acidity

*Same, for the 3 Acidity classes. Even more skewed (79% High) — class-weighted training compared against unweighted.*

### 4.3 Harmonize food-pairings

*Per-label F1 across the top-N Harmonize pairings, plus Hamming loss. A handful of example wines with their predicted vs. actual pairing vectors.*

---

## 5. SHAP analysis

*Top-20 features by mean absolute SHAP on the rating model, plus 3–4 dependence plots for the most interesting terroir variables (candidates: GDD, harvest-month precip, calcareous flag, diurnal range). Figures live under `reports/figures/` and are embedded here at generation time.*

---

## 6. Drink-now and age-well rankings

### 6.1 Drink-now (opening year 2026)

For each major grape, the top-N monogrape wines predicted to drink best in 2026. Default filter: `--max-vintage-age 5` (fresh-style only) for whites + aromatic reds; no age cap for cellar-style reds.

*Tables per grape (Cabernet Sauvignon, Pinot Noir, Chardonnay, Riesling, Nebbiolo, Tempranillo, Syrah, Sangiovese — list finalised at generation), each with columns: WineryName, WineName, RegionName, Vintage, predicted_rating, confidence band. CLI command that produced each table cited above it.*

### 6.2 Age-well (opening years 2026 → 2036)

For each major grape, the top-N monogrape wines whose predicted-rating trajectory still rises or peaks late within the 10-year horizon.

*Tables per grape with columns: WineryName, WineName, RegionName, Vintage, predicted_peak_year, predicted_peak_rating, slope_to_peak. Rows where `age_at_review` had to be clipped to the training range are flagged.*

### 6.3 Standout wines of the year (2026 → 2031)

One curated shortlist per drinking year across the next five years — the wines the model predicts will be at their best *in that specific year*. Produced by `vininator recommend standout-years --from-year 2026 --to-year 2031`.

*Six tables (2026, 2027, 2028, 2029, 2030, 2031), each top-10 monogrape, with columns: WineryName, WineName, RegionName, Vintage, predicted_rating, confidence band. A wine may appear in more than one year's list when its projected drink-now trajectory plateaus; that's expected and noted inline.*

### 6.4 Overperformer outliers (more special than expected)

Wines predicted to outscore their peer-group baseline (per-`(GrapeMajority, RegionName)` and per-`(RegionName, Vintage)` means, training-fold only) by the largest margin — and where the lower confidence bound still clears that baseline, so the surprise isn't an artefact of a wide prediction interval. These are the "punching above their weight" picks, deliberately *not* the highest absolute ratings (which skew to famous producers). Produced by `vininator recommend outliers --opening-year 2026`.

*Table: WineryName, WineName, RegionName, Vintage, predicted_rating, peer_baseline, overperformance (= predicted − baseline), confidence band, sorted by overperformance descending. Short commentary on what the model thinks makes each outlier special — terroir-driven (a standout vintage in a modest region) vs. structure-driven — read off the SHAP contributions for the top few.*

### 6.5 Caveats

- Rankings are conditional on wines *in X-Wines*. Not a ranking of the entire wine world.
- Producer effects dominate — expect lists to skew toward well-rated wineries. That's signal, not bug, but worth knowing.
- Aged-wine projections beyond ~10 years post-vintage are extrapolation; clipped rows are flagged.
- Climate is region-centroid, not vineyard-parcel. See the disclaimer block in the README for the full list of scoping decisions.
- Overperformer outliers are only as trustworthy as the baseline they're measured against — sparse `(RegionName, Vintage)` cells make for noisy baselines, so the outlier table is restricted to peer groups with enough support (threshold set in config).

---

## 7. Qualitative sanity check

*The 10 known-wines exercise from Phase 5: hand-picked wines I personally know, model's predictions for rating / Body / Acidity / pairings, and short commentary on agreements and disagreements. The disagreements are the interesting half.*

---

## 8. Limitations & caveats

Pulled from the README disclaimer block plus model-specific issues surfaced during evaluation. Generated at build time so the list stays in sync with the README.

---

## 9. Reproduction

Exact CLI sequence to rebuild every artifact in this document from a fresh clone. Includes data download, geocoding, NASA POWER + SoilGrids pulls, feature assembly, model training, ablations, recommender runs, and the final `scripts/build_results.py` invocation that emits this file.

---

## Acknowledgements

- **X-Wines dataset** — Xavier 2023, MDPI BDCC. CC0 1.0.
- **NASA POWER** — LaRC POWER Project. Underlying: MERRA-2 + CERES SYN1DEG.
- **SoilGrids** — ISRIC. Hengl et al., 2021.
