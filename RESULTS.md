# Vininator 3000 — Results

> **Status:** placeholder. Phase 4 (modeling) is still in progress. This file is regenerated end-to-end by `scripts/build_results.py` once the trained models, ablation runs, and recommender outputs exist. The structure below is the contract — sections will be populated with real numbers, tables, and figures as each phase completes. Do not hand-edit; edit the generator.

---

## 1. Headline: the picks

*To be filled by `scripts/build_results.py`.*

The wines the models actually surface — a teaser of the top drink-now bottles for the current year, the top cellar candidates, and the most striking overperformers — with one paragraph on how trustworthy the lists are: the rating model's cell-level RMSE against the best leakage-safe baseline, in one number. The full tables live in §3–4; the full quality story in §5.

---

## 2. Setup

- **Dataset variant:** *(test | slim | full — filled at generation)*
- **Train / test / future-vintage split sizes:** *(filled)*
- **Seed:** *(filled)*
- **Git SHA:** *(filled)*
- **Experiment tracking run:** *(MLflow link, filled)*
- **Reproduction command:** see [§10](#10-reproduction).

---

## 3. Drink-now and age-well rankings

### 3.1 Drink-now (opening year 2026)

For each major grape, the top-N monogrape wines predicted to drink best in 2026. Default filter: `--max-vintage-age 5` (fresh-style only) for whites + aromatic reds; no age cap for cellar-style reds.

*Tables per grape (Cabernet Sauvignon, Pinot Noir, Chardonnay, Riesling, Nebbiolo, Tempranillo, Syrah, Sangiovese — list finalised at generation), each with columns: WineryName, WineName, RegionName, Vintage, predicted_rating, confidence band, predicted body/acidity, top pairings. CLI command that produced each table cited above it.*

### 3.2 Age-well (opening years 2026 → 2036)

For each major grape, the top-N monogrape wines whose predicted-rating trajectory still rises or peaks late within the 10-year horizon.

*Tables per grape with columns: WineryName, WineName, RegionName, Vintage, predicted_peak_year, predicted_peak_rating, slope_to_peak. Rows where `age_at_review` had to be clipped to the training range are flagged.*

---

## 4. Standouts and overperformers

### 4.1 Standout wines of the year (2026 → 2031)

One curated shortlist per drinking year across the next five years — the wines the model predicts will be at their best *in that specific year*. Produced by `vininator recommend standout-years --from-year 2026 --to-year 2031`.

*Six tables (2026, 2027, 2028, 2029, 2030, 2031), each top-10 monogrape, with columns: WineryName, WineName, RegionName, Vintage, predicted_rating, confidence band. A wine may appear in more than one year's list when its projected drink-now trajectory plateaus; that's expected and noted inline.*

### 4.2 Overperformer outliers (more special than expected)

Wines predicted to outscore their peer-group baseline (per-`(GrapeMajority, RegionName)` and per-`(RegionName, Vintage)` means, training-fold only) by the largest margin — and where the lower confidence bound still clears that baseline, so the surprise isn't an artefact of a wide prediction interval. These are the "punching above their weight" picks, deliberately *not* the highest absolute ratings (which skew to famous producers). Produced by `vininator recommend outliers --opening-year 2026`.

*Table: WineryName, WineName, RegionName, Vintage, predicted_rating, peer_baseline, overperformance (= predicted − baseline), confidence band, sorted by overperformance descending. Short commentary on what the model thinks makes each outlier special — terroir-driven (a standout vintage in a modest region) vs. structure-driven — read off the SHAP contributions for the top few.*

### 4.3 Ranking caveats

- Rankings are conditional on wines *in X-Wines*. Not a ranking of the entire wine world.
- Producer effects dominate — expect lists to skew toward well-rated wineries. That's signal, not bug, but worth knowing.
- Aged-wine projections beyond ~10 years post-vintage are extrapolation; clipped rows are flagged.
- Climate is region-centroid, not vineyard-parcel. See the disclaimer block in the README for the full list of scoping decisions.
- Overperformer outliers are only as trustworthy as the baseline they're measured against — sparse `(RegionName, Vintage)` cells make for noisy baselines, so the outlier table is restricted to peer groups with enough support (threshold set in config).

---

## 5. Rating model quality

How seriously to take §1–4. All rating tables report two levels: per-rating RMSE/MAE (comparable to the PROJECT.md baseline numbers, floored by the within-cell spread of user opinions — the `noise_floor` row) and cell-level weighted RMSE/MAE (predicted vs. observed mean rating per `(wine, vintage, age)` cell — the number that measures ranking quality, where user noise is averaged out).

### 5.1 Held-out wines (random `WineID` split)

*Table: per-rating and cell-level RMSE + MAE for the trained model and the baseline grid — global mean, per-`WineryID` mean, per-`(RegionName, Vintage)` mean, per-`(GrapeMajority, RegionName)` mean — plus the `noise_floor` for the per-rating columns.*

### 5.2 Future-vintage holdout (train ≤ 2018, test 2019–2021)

*Same metrics, on the vintage-generalization split. Interpretation note (rendered with the table): this split contains wines seen in training — only the vintage is new — so its RMSE is expected to be lower than §5.1's and the two are not comparable to each other. §5.1 asks "does the model generalize to a new wine"; this section asks "does it generalize to a new year of a known wine".*

### 5.3 Confidence intervals

*Per-prediction lo / hi from the quantile heads — bands on the wine-vintage mean rating, not on individual user ratings — summarized as coverage of observed cell means on the held-out set. These bands gate the overperformer table (§4.2).*

---

## 6. Profile + Harmonize models

Quality of the enrichment columns in the ranking tables (predicted body, acidity, and pairings). All metrics count each held-out wine-vintage once (not once per rating), matching how the models train — per-rating metrics would be dominated by popular wines.

### 6.1 Body

*Confusion matrix and per-class F1 against the 5 X-Wines Body classes. Macro-F1 reported headline, not accuracy — the class skew (44% Full-bodied) makes accuracy uninformative.*

### 6.2 Acidity

*Same, for the 3 Acidity classes. Even more skewed (79% High) — class-weighted training compared against unweighted.*

### 6.3 Harmonize food-pairings

*Per-label F1 across the top-N Harmonize pairings, plus Hamming loss. A handful of example wines with their predicted vs. actual pairing vectors.*

---

## 7. Diagnostics: SHAP + ablations

What each feature block actually earns. Terroir is a working assumption of the feature design (see PROJECT.md §1), so this section is where that assumption gets audited — reported honestly even if the answer is "not much".

### 7.1 SHAP analysis

*Top-20 features by mean absolute SHAP on the rating model, plus 3–4 dependence plots for the most interesting terroir variables (candidates: GDD, harvest-month precip, calcareous flag, diurnal range). Figures live under `reports/figures/` and are embedded here at generation time.*

### 7.2 Ablations

*Table with rows = ablated block (none / − terroir / − producer / − `age_at_review`) and columns = cell-level RMSE on each split, plus the delta against the full model. RMSE head only (quantile heads don't affect the conclusion).*

---

## 8. Qualitative sanity check

*The 10 known-wines exercise from Phase 5: hand-picked wines I personally know, model's predictions for rating / Body / Acidity / pairings, and short commentary on agreements and disagreements. The disagreements are the interesting half.*

---

## 9. Limitations & caveats

Pulled from the README disclaimer block plus model-specific issues surfaced during evaluation. Generated at build time so the list stays in sync with the README.

---

## 10. Reproduction

Exact CLI sequence to rebuild every artifact in this document from a fresh clone. Includes data download, geocoding, NASA POWER + SoilGrids pulls, feature assembly, model training, ablations, recommender runs, and the final `scripts/build_results.py` invocation that emits this file.

---

## Acknowledgements

- **X-Wines dataset** — Xavier 2023, MDPI BDCC. CC0 1.0.
- **NASA POWER** — LaRC POWER Project. Underlying: MERRA-2 + CERES SYN1DEG.
- **SoilGrids** — ISRIC. Hengl et al., 2021.
