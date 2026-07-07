"""Single source of truth for filesystem paths and external-service config.

All other modules import the resolved `settings` singleton — they never
construct paths or read environment variables themselves. This keeps the
layout decision in one place and makes tests trivial: override `DATA_DIR`
via env or fixture and everything downstream follows.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

XWINES_GITHUB_RAW = "https://raw.githubusercontent.com/rogerioxavier/X-Wines/main/Dataset/last"

# Filename templates per variant. The slim/full variants live on the project's
# Google Drive (not in-repo); the test variant ships directly from GitHub.
XWINES_VARIANTS: dict[str, dict[str, str]] = {
    "test": {
        "wines_csv": "XWines_Test_100_wines.csv",
        "ratings_csv": "XWines_Test_1K_ratings.csv",
    },
    "slim": {
        "wines_csv": "XWines_Slim_1K_wines.csv",
        "ratings_csv": "XWines_Slim_150K_ratings.csv",
    },
    "full": {
        "wines_csv": "XWines_Full_100K_wines.csv",
        "ratings_csv": "XWines_Full_21M_ratings.csv",
    },
}

XWINES_WINES_PARQUET = "xwines_wines.parquet"
XWINES_RATINGS_PARQUET = "xwines_ratings.parquet"

# Geocoding — Nominatim asks for a contactable user agent in their TOS and caps
# unauthenticated traffic at 1 req/sec; both are non-negotiable. The default
# socket timeout (1s) is too aggressive for the free tier — bump to 10s so a
# slow response cleanly exhausts the RateLimiter's retries instead of failing
# the whole row.
#
# The contact string is supplied via env (`VININATOR_NOMINATIM_CONTACT`) and
# has no in-source default.
NOMINATIM_USER_AGENT_TEMPLATE = "vininator-3000/0.1 ({contact})"
NOMINATIM_RATE_LIMIT_SEC = 1.0
NOMINATIM_TIMEOUT_SEC = 10.0

# Flush the geocode cache every N rows so a Ctrl+C 30 minutes into a 36-minute
# cold run loses at most N rows of work, not the whole run.
GEOCODE_CHECKPOINT_EVERY = 25

# Outer backoff when Nominatim returns 429 and the RateLimiter exhausts its own
# retries. Pause increasingly long between offending rows so we stop hammering
# the server while still making progress when the cool-down clears.
GEOCODE_BACKOFF_BASE_SEC = 30.0
GEOCODE_BACKOFF_CAP_SEC = 300.0

GEOCODE_PARQUET = "geocode.parquet"

# Geocode `result_type` blacklist — applied at read time by `filter_to_usable`.
# Audit of the real 2160-region geocode showed that Nominatim's `result_type`
# is wildly inconsistent for wine regions: many real appellations come back
# tagged as "restaurant", "residential", "river", "volcano", "peak", etc. A
# whitelist would drop hundreds of real wine regions. So we blacklist only
# entity types that are unambiguously non-region (transport stops, postal
# infrastructure, financial POIs, healthcare, fuel/parking) and keep the long
# tail. Some bad rows (Buenos Aires-style city centroids) still slip through;
# the downstream null detection in `compute_soil_features` catches them.
GEOCODE_BAD_RESULT_TYPES: frozenset[str] = frozenset(
    {
        "bus_stop",
        "station",
        "platform",
        "halt",
        "post_office",
        "post_box",
        "atm",
        "bank",
        "fuel",
        "hospital",
        "clinic",
        "school",
        "college",
        "parking",
        "elevator",
        "fire_station",
    }
)

# ---------------------------------------------------------------------------
# Soil + DEM
# ---------------------------------------------------------------------------
#
# SoilGrids v2 (ISRIC) is free, no auth, but the public endpoint is flaky on
# single-pixel queries. We buffer the centroid into a 3x3 grid (~500 m square)
# and average the per-property `mean` values across the 9 points so that one
# bad pixel can't flip drainage_class between runs.

SOILGRIDS_BASE_URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"

# SoilGrids does NOT ship a 0-30cm aggregate. It exposes three topsoil bands
# that we depth-weight-average to get a single value per property:
#   0-5cm   thickness 5  → weight 5/30
#   5-15cm  thickness 10 → weight 10/30
#   15-30cm thickness 15 → weight 15/30
SOILGRIDS_DEPTHS: tuple[str, ...] = ("0-5cm", "5-15cm", "15-30cm")
SOILGRIDS_DEPTH_WEIGHTS: dict[str, float] = {
    "0-5cm": 5.0 / 30.0,
    "5-15cm": 10.0 / 30.0,
    "15-30cm": 15.0 / 30.0,
}

SOILGRIDS_PROPERTIES: tuple[str, ...] = (
    "phh2o",
    "cec",
    "clay",
    "sand",
    "silt",
    "soc",
    "bdod",
    "cfvo",
)
SOILGRIDS_BUFFER_DEG = 0.005
SOILGRIDS_RETRIES = 3
SOILGRIDS_BACKOFF_SEC: tuple[float, ...] = (1.0, 4.0, 16.0)
SOILGRIDS_TIMEOUT_SEC = 30.0
SOILGRIDS_RATE_LIMIT_SEC = 1.0

# SoilGrids v2 returns d-factored integers — `unit_measure.d_factor` in the
# response. Divide the raw `mean` by this number to land directly in the layer's
# `target_units` (NOT an intermediate `mapped_units`). Values verified against
# the live response metadata for Tuscany (43.46, 11.04).
#
# clay / sand / silt:  d_factor=10  → %
# phh2o / cec / soc:   d_factor=10  → pH / cmol(c)/kg / g/kg
# bdod:                d_factor=100 → kg/dm³
# cfvo:                d_factor=10  → cm³/100cm³ (vol %)
#
# Tuple is (final_divisor, target_unit_label) per property.
SOILGRIDS_UNIT_CONVERSIONS: dict[str, tuple[float, str]] = {
    "phh2o": (10.0, "pH"),
    "clay": (10.0, "%"),
    "sand": (10.0, "%"),
    "silt": (10.0, "%"),
    "soc": (10.0, "g/kg"),
    "cec": (10.0, "cmol(c)/kg"),
    "bdod": (100.0, "kg/dm³"),
    "cfvo": (10.0, "vol %"),
}

# Open-Elevation: free, no auth, intermittently down. One POST per region
# returns elevations for our 3x3 grid; slope is then derived in-process via
# Horn's algorithm. The grid spacing (~300 m at the equator) matches SRTM's
# 30 m source resolution closely enough that slope is meaningful but not noisy.
OPEN_ELEVATION_URL = "https://api.open-elevation.com/api/v1/lookup"
OPEN_ELEVATION_TIMEOUT_SEC = 30.0
OPEN_ELEVATION_RATE_LIMIT_SEC = 1.0
DEM_GRID_DEG = 0.0027
DEM_LAT_METERS_PER_DEG = 111_320.0

# Soil feature derivations.
CALCAREOUS_PH_THRESHOLD = 7.5
DRAINAGE_CLAY_PCT = 40.0
DRAINAGE_SAND_PCT = 60.0

SOIL_RAW_DIRNAME = "soil_raw"
DEM_RAW_DIRNAME = "dem"
SOIL_PARQUET = "soil.parquet"

# ---------------------------------------------------------------------------
# Climate (NASA POWER Daily API)
# ---------------------------------------------------------------------------
#
# NASA POWER serves daily meteorological data derived from MERRA-2 (temperature,
# precipitation) and CERES SYN1DEG (solar radiation), at ~0.5° / ~55 km grid
# resolution. It exposes a plain JSON REST endpoint, no account, no token, and
# a single request per region returns every year in `CLIMATE_YEAR_RANGE` for
# all five daily variables. One JSON cache file per region under
# `data/raw/nasa_power/{slug}.json` *is* the resume state — restart picks up
# wherever the disk says it left off.
#
# We pivoted off Open-Meteo (ERA5-Land, 0.1°) because its free tier turned out
# to bill *per data point* — 30 years × 5 variables × one coordinate exceeded
# the daily quota in a single region pull. NASA POWER's coarser resolution is
# the cost of staying free; for the growing-degree-day / heat-spike /
# harvest-precip features we actually compute, the ~55 km vs ~11 km gap is
# negligible compared to the centroid-vs-vineyard error already baked into
# region geocoding.

NASA_POWER_BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
# Agroclimatology community — chooses units (temperatures in °C, solar in
# MJ/m²/day) appropriate for viticulture features.
NASA_POWER_COMMUNITY = "AG"

# NASA POWER variable IDs → internal feature column names (see
# `load_nasa_power_daily`). Units verified via the live API metadata:
#   T2M               → tmean_c   (Temperature at 2 Meters, °C)
#   T2M_MIN           → tmin_c    (Temperature at 2 Meters Min, °C)
#   T2M_MAX           → tmax_c    (Temperature at 2 Meters Max, °C)
#   PRECTOTCORR       → precip_mm (Precipitation Corrected, mm/day)
#   ALLSKY_SFC_SW_DWN → ssrd_mj   (All Sky Surface Shortwave Downward, MJ/m²/day)
# Units land in exactly the form `compute_climate_features` expects — no
# downstream unit conversion required.
NASA_POWER_DAILY_VARS: tuple[str, ...] = (
    "T2M",
    "T2M_MIN",
    "T2M_MAX",
    "PRECTOTCORR",
    "ALLSKY_SFC_SW_DWN",
)

# POWER returns -999.0 for days where source data was unavailable. The parser
# coerces this sentinel to null so `compute_climate_features` flags affected
# growing seasons as `is_partial` the same way Open-Meteo's explicit nulls did.
NASA_POWER_FILL_VALUE = -999.0

NASA_POWER_TIMEOUT_SEC = 1800.0
# POWER doesn't publish a hard rate limit, but the API docs warn that
# hammering the same coordinate may result in opaque blocking. 1 s spacing
# keeps the 1,377-region pull at ~23 minutes wall time and well clear of
# "hammering" territory.
NASA_POWER_RATE_LIMIT_SEC = 1.0
# Backoff for transient errors (5xx, network blips). POWER does not return a
# structured `Retry-After` header, so every retry uses the same exponential
# schedule — no special-cased 429 path like Open-Meteo had.
NASA_POWER_BACKOFF_SEC: tuple[float, ...] = (2.0, 8.0, 30.0, 60.0, 120.0, 300.0)

# Attribution baked into the parquet metadata. POWER data are in the public
# domain but the project requests acknowledgement of LaRC and the underlying
# data sources (MERRA-2, CERES SYN1DEG).
NASA_POWER_ATTRIBUTION = (
    "Weather data from the NASA Langley Research Center (LaRC) POWER Project, "
    "funded through the NASA Earth Science Directorate Applied Science Program. "
    "Underlying sources: MERRA-2 (meteorology) and CERES SYN1DEG (solar radiation)."
)

NASA_POWER_RAW_DIRNAME = "nasa_power"
CLIMATE_PARQUET = "climate.parquet"
CLIMATOLOGY_PARQUET = "climatology.parquet"
TERROIR_PARQUET = "terroir.parquet"

# 1991 is the start of the climate record we care about (also the start of
# the WMO-era ERA5 baseline window the literature uses); 2021 is the X-Wines
# Date cutoff. Anything outside this window is out of scope.
CLIMATE_YEAR_RANGE: tuple[int, int] = (1991, 2021)

# Climatology baseline ends at the training cutoff (2018) so the anomaly column
# carries zero information leakage into the 2019–2021 future-vintage holdout.
# This deviates from the WMO standard (1991–2020); the leakage cost outweighs
# the textbook convention.
CLIMATOLOGY_WINDOW: tuple[int, int] = (1991, 2018)
CLIMATOLOGY_MIN_YEARS = 20

# Growing-season month windows: month numbers (inclusive).
# NH: April–October of vintage_year.
# SH: October–December of (vintage_year − 1) plus January–April of vintage_year.
# Vintage year = harvest year for both hemispheres (X-Wines convention).
GROWING_SEASON_NH: tuple[int, int] = (4, 10)
GROWING_SEASON_SH_PREV_YEAR: tuple[int, int] = (10, 12)
GROWING_SEASON_SH_VINTAGE: tuple[int, int] = (1, 4)

# Spring frost window: the last two months before bud break in each hemisphere.
SPRING_FROST_MONTHS_NH: tuple[int, int] = (4, 5)
SPRING_FROST_MONTHS_SH: tuple[int, int] = (10, 11)

GDD_BASE_TEMP_C = 10.0
HEAT_SPIKE_TMAX_C = 35.0
FROST_TMIN_C = 0.0
HARVEST_WINDOW_DAYS = 30

# Absolute climate features. Anomaly columns are `<name>_anom`.
CLIMATE_ABSOLUTE_FEATURES: tuple[str, ...] = (
    "gdd_10c",
    "precip_total_mm",
    "precip_harvest_mm",
    "heat_spike_days",
    "frost_days_spring",
    "diurnal_range_mean",
    "solar_total_mj",
)

# ---------------------------------------------------------------------------
# Phase 3 — Feature assembly & splits
# ---------------------------------------------------------------------------

# Future-vintage holdout: train on vintages ≤ 2018, hold out 2019–2021 as a
# separate test to measure whether the model learned terroir or memorised
# region averages. This window matches the training cutoff used for the
# climatology baseline so the anomaly columns carry zero leakage.
FUTURE_VINTAGE_HOLDOUT: tuple[int, int] = (2019, 2021)

# Fraction of wines (by WineID) held back as the random test set. The
# future-vintage holdout is separate and additional.
TRAIN_HOLDOUT_FRAC = 0.15

# Seed for the WineID shuffle that produces train vs. test wines. Fix this
# once and never change it — changing after training makes comparisons
# meaningless.
WINE_SPLIT_SEED = 42

# Vocabulary sizes for multi-hot features derived from list columns.
# Computed on the training fold only; unseen grapes/pairings → all-zero row.
TOP_K_GRAPES = 50
TOP_K_HARMONIZE = 30

PROCESSED_TRAIN_PARQUET = "train.parquet"
PROCESSED_TEST_PARQUET = "test.parquet"
PROCESSED_FUTURE_VINTAGE_TEST_PARQUET = "future_vintage_test.parquet"

# Phase 5 evaluation artifacts (consumed by scripts/build_results.py).
ABLATIONS_PARQUET = "ablations.parquet"
SHAP_IMPORTANCE_PARQUET = "shap_importance.parquet"

# SHAP runs on a seeded sample of eval cells — CatBoost's ShapValues over the
# full test split (~340k cells × 121 features) costs multiple GB of float64
# for no ranking benefit; 100k cells gives stable mean-|SHAP| orderings.
SHAP_SAMPLE_CELLS = 100_000
SHAP_TOP_N_FEATURES = 20
# Terroir variables whose SHAP dependence plots RESULTS.md §7 wants to show.
SHAP_DEPENDENCE_FEATURES: tuple[str, ...] = (
    "gdd_10c",
    "gdd_10c_anom",
    "precip_harvest_mm",
    "calcareous",
)

# ---------------------------------------------------------------------------
# Phase 4 — Modeling
# ---------------------------------------------------------------------------
#
# These constants describe how the 127-column processed table (see
# features/build.py) is sliced into model inputs. `models/dataset.py` is the
# single consumer — it never re-derives the feature split, so changing the
# contract here changes it for the rating, profile, and harmonize models at
# once. Grape/pair multi-hot columns are discovered dynamically from the
# parquet schema (their count is data-driven), so they are not listed here.

# Target columns, one per modeling task.
RATING_TARGET = "rating"
PROFILE_TARGETS: tuple[str, ...] = ("body_label", "acidity_label")
HARMONIZE_TARGET_PREFIX = "pair_"

# Aggregation keys (models/dataset.py). Every feature is either wine-level,
# (region, vintage)-level, or the age itself — so rating rows sharing these
# keys carry identical feature vectors and can be collapsed to one weighted
# row without changing the weighted-RMSE optimum. The full variant shrinks
# ~7× (15.5M rows → 2.25M cells), which is what makes full-data training fit
# in 16 GB of RAM.
RATING_CELL_KEYS: tuple[str, ...] = ("wine_id", "vintage_year", "age_at_review")
# Wine-level targets (body, acidity, pair_*) are constant per wine; their
# trainers collapse to one row per (wine, vintage) and drop the age column.
WINE_VINTAGE_KEYS: tuple[str, ...] = ("wine_id", "vintage_year")
AGE_COL = "age_at_review"
# Ratings collapsed into a cell — carried as metadata (never a feature: it is
# popularity, unknowable for unseen wines) and used to weight cell-level eval.
CELL_N_RATINGS_COL = "n_ratings_cell"

# Columns that are never features for any model: row identity, the split tag,
# the loss weight, the review timestamp, the aggregation cell count, and the
# rating itself (rating is only ever a target). Per-model target columns are
# excluded on top of these.
MODEL_METADATA_COLS: tuple[str, ...] = (
    "rating_id",
    "wine_id",
    "rating_date",
    "split",
    "sample_weight",
    "rating",
    CELL_N_RATINGS_COL,
)

# Producer aggregate features (train-fold, leave-one-wine-out; see
# features/build.py). Named here so the Phase 5 ablation can drop the block.
PRODUCER_FEATURE_COLS: tuple[str, ...] = (
    "producer_mean_rating",
    "producer_rating_std",
    "producer_n_reviews",
)

# The per-row CatBoost weight column (log(1 + n_ratings_in_train), train-fold).
SAMPLE_WEIGHT_COL = "sample_weight"
# Grouping key for the early-stopping validation fold — split by wine, never
# by row, to mirror the train/test boundary and avoid leakage.
GROUP_KEY_COL = "wine_id"

# Categorical features handed to CatBoost via `cat_features`. CatBoost rejects
# NaN in categoricals, so `models/dataset.py` fills nulls with a sentinel and
# casts high-cardinality integer keys (winery_id) to string first.
CATEGORICAL_FEATURE_COLS: tuple[str, ...] = (
    "wine_type",
    "country",
    "region_name",
    "winery_id",
    "grape_majority",
    "drainage_class",
    "body_label",
    "acidity_label",
)
CATEGORICAL_NULL_SENTINEL = "unknown"

# Boolean features cast to Int8 0/1 and treated as numerical (CatBoost handles
# NaN in numerics natively, so a missing region's flags stay null).
BOOL_FEATURE_COLS: tuple[str, ...] = ("is_partial", "calcareous")

# Quantile heads for the rating model → the recommender's _lo / _hi bands.
RATING_QUANTILES: tuple[float, float] = (0.1, 0.9)

# Fraction of TRAIN wines carved out (by wine_id) for early-stopping eval.
GROUP_VAL_FRAC = 0.1
# Seed for the grouped validation shuffle and CatBoost's RNG. Fix once.
MODEL_SEED = 42

# MLflow experiment name; the tracking store is the gitignored repo-root
# `mlruns/` directory (see Settings.mlflow_tracking_dir).
MLFLOW_EXPERIMENT = "vininator"
MLFLOW_RUNS_DIRNAME = "mlruns"

# Saved model bundle stems under data/models/ (one `.cbm` + one `.meta.json`
# each). The recommender (Phase 6) loads these by name.
MODELS_DIRNAME = "models"
RATING_BUNDLE = "rating"
RATING_QUANTILE_LO_BUNDLE = "rating_q_lo"
RATING_QUANTILE_HI_BUNDLE = "rating_q_hi"
BODY_BUNDLE = "body"
ACIDITY_BUNDLE = "acidity"
HARMONIZE_BUNDLE = "harmonize"

# ---------------------------------------------------------------------------
# Phase 6 — Recommender
# ---------------------------------------------------------------------------
#
# The recommender loads the saved bundles and the already-built processed
# parquets, then sweeps `age_at_review` (opening_year − vintage_year) against
# the rating model to project drink-now / age-well rankings. Vintage — and
# therefore terroir — is held constant, so there is no live fetch and no
# feature re-engineering (the same rule `eval/sanity.py` follows).

# Default opening year for drink-now / outliers when the CLI omits `--opening-year`.
DEFAULT_OPENING_YEAR = 2026
# Age-well sweeps opening_year .. opening_year + horizon (inclusive).
RECOMMEND_HORIZON_YEARS = 10
# Standout-of-the-year window (inclusive), one shortlist per year.
STANDOUT_YEAR_RANGE: tuple[int, int] = (2026, 2031)
# Default top-N sizes: the long rankings keep everything, these bound the
# curated / printed shortlist.
RECOMMEND_TOP_N = 50
STANDOUT_TOP_N = 10
# Pairings shown per wine in the ranking tables (highest-probability first).
RECOMMEND_TOP_PAIRINGS = 5
# |Δrating| below which an age-well trajectory counts as "holds steady".
AGE_WELL_PLATEAU_EPS = 0.02

# Overperformer outliers: a peer group (per-(grape, region) or per-(region,
# vintage)) is only trusted as a baseline when it has at least this many
# distinct training wines — matches the Phase 1 EDA "cells with >= 5 wines"
# convention. `peer_baseline` aggregates the supported group means by max
# (the stricter, harder-to-beat bar) or mean.
OUTLIER_MIN_PEER_WINES = 5
OUTLIER_PEER_AGG: Literal["max", "mean"] = "max"

# ---------------------------------------------------------------------------
# Recommendation library (scripts/build_library.py)
# ---------------------------------------------------------------------------
#
# A browsable per-grape catalog under reports/recommendations/: top-N lists for
# recent wines (drink-now, age-well, future greats, best value), grouped by
# wine type, with expanded lists for the favorite wines. Reuses the Phase 6
# recommenders unchanged — these constants only pick what the pages show.

# Exclusive floor: the library covers vintage_year > this. Set to 2018 (→ 2019–
# 2021) after auditing the vintage skew. Scored at a fixed age the vintages
# 2017–2021 are flat (see best-years.md "Best vintages"), so 2017/2018 dominating
# the drink-now lists was purely an age artifact: at opening year 2026 an older
# vintage is swept to a higher age and the model's small positive age slope lifts
# it. Restricting to 2019+ keeps the swept-age spread (5–7 yrs) narrow.
LIBRARY_MIN_VINTAGE = 2018
LIBRARY_WINE_TYPES: tuple[str, ...] = ("Red", "White", "Rosé")
LIBRARY_MIN_BOTTLES_PER_GRAPE = 100
LIBRARY_TOP_N = 10
LIBRARY_FUTURE_PEAK_MIN_YEAR = 2030  # "future great" = peaks this year or later

# Favorites get deeper lists. Grape favorites are monogrape slices; region
# favorites are wine *styles* (blends defined by their region — Amarone is a
# Corvina-based blend, Super Tuscans are Toscana IGT blends), sliced by
# region_name with the monogrape filter off, Red only.
LIBRARY_FAVORITE_TOP_N = 25
LIBRARY_FAVORITE_GRAPES: tuple[str, ...] = ("Sangiovese", "Primitivo", "Nero d'Avola")
LIBRARY_FAVORITE_REGIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "Amarone della Valpolicella",
        (
            "amarone della valpolicella",
            "amarone della valpolicella classico",
            "amarone della valpolicella valpantena",
        ),
    ),
    (
        "Super Tuscan (Toscana IGT)",
        ("toscana", "maremma toscana", "colli della toscana centrale", "costa toscana"),
    ),
)

# Best-vintages table: compare vintages at the same age so the age effect
# doesn't confound vintage quality (a 2017 is 9 years old in 2026, a 2021 five).
LIBRARY_VINTAGE_QUALITY_AGE = 5

# ---------------------------------------------------------------------------
# Price metadata (features/price.py)
# ---------------------------------------------------------------------------
#
# Price is post-hoc catalog metadata, never a model feature — it joins onto the
# scored candidate table at library-build time. Source: Kaggle "Wine Reviews"
# (zynicide) 2017 snapshot, prices in USD, CC BY-NC-SA 4.0 (non-commercial).
# Static CSV, manual drop like the slim/full X-Wines variants — no auth-gated
# fetch code.

WINE_REVIEWS_DIRNAME = "wine_reviews"  # under data/raw/
WINE_REVIEWS_CSV = "winemag-data-130k-v2.csv"
WINE_REVIEWS_URL = "https://www.kaggle.com/datasets/zynicide/wine-reviews"
PRICE_PARQUET = "price.parquet"  # under data/interim/
PRICE_SOURCE = "wine-reviews-2017"
PRICE_SOURCE_CURRENCY = "USD"  # currency stored in price.parquet (source-native)
PRICE_MATCH_CONFIDENCES: tuple[str, ...] = ("exact", "winery-median", "none")
PRICE_MIN_WINERY_ROWS = 3  # winery-median fallback needs at least this many priced rows

# Value views are shown in EUR (the maintainer thinks in euros). Source prices
# stay USD in the parquet; join_price converts with this FX assumption. The 2017
# snapshot is old, so this is a rough constant, not a live rate — stated openly.
PRICE_DISPLAY_CURRENCY = "EUR"
PRICE_USD_PER_EUR = 1.08  # USD per 1 EUR; price_eur = price_usd / PRICE_USD_PER_EUR

# The snapshot is a 2017 price list, but the library values everything as of the
# opening year (DEFAULT_OPENING_YEAR, 2026). join_price bridges the gap with two
# compounding adjustments, applied at join time because they need the per-row
# vintage that price.parquet doesn't carry:
#   1. Inflation — general USD price rise from the snapshot year to the opening
#      year (uniform across wines). ~3%/yr ≈ +30% over 2017→2026.
#   2. Aging premium — a bottle gets pricier as it ages; applied over the bottle's
#      age in the opening year (older vintages cost more), capped so ancient
#      vintages don't extrapolate to absurd prices.
PRICE_SNAPSHOT_YEAR = 2017
PRICE_INFLATION_ANNUAL = 0.03  # approx. US CPI averaged over 2017–2025
PRICE_APPRECIATION_ANNUAL = 0.04  # fine-wine aging premium per year of bottle age
PRICE_APPRECIATION_MAX_YEARS = 15  # cap the aging exponent; beyond this is guesswork

# value_score = predicted_rating / price_eur (rating per euro). Price bands are
# (label, upper bound in EUR, exclusive); above the last bound = cult.
VALUE_PRICE_CAP_EUR = 40.0  # the library's per-grape "best under a cap" table
PRICE_BANDS: tuple[tuple[str, float], ...] = (("budget", 15.0), ("mid", 40.0), ("premium", 150.0))
PRICE_BAND_TOP = "cult"

# Recommendation output parquets under data/processed/ (consumed by
# scripts/build_results.py for the RESULTS.md ranking tables).
RECOMMENDATIONS_DRINK_NOW_PARQUET = "recommendations_drink_now.parquet"
RECOMMENDATIONS_AGE_WELL_PARQUET = "recommendations_age_well.parquet"
RECOMMENDATIONS_AGE_WELL_SUMMARY_PARQUET = "recommendations_age_well_summary.parquet"
RECOMMENDATIONS_STANDOUT_YEARS_PARQUET = "recommendations_standout_years.parquet"
RECOMMENDATIONS_OUTLIERS_PARQUET = "recommendations_outliers.parquet"

# ---------------------------------------------------------------------------
# Phase 7 — Findings publication (scripts/build_results.py)
# ---------------------------------------------------------------------------
#
# The generator reads the trained bundles + recommendation parquets and emits
# RESULTS.md end-to-end (no hand-typed numbers). These constants pick the grapes
# and wines it reports on; per-grape ranking parquets land in reports/tables/.

# Per-grape ranking tables in RESULTS.md §3: (CLI slug, fresh-style age cap).
# None = no cap (cellar-style reds); the aromatic whites get a 5-year cap.
RESULTS_MAJOR_GRAPES: tuple[tuple[str, int | None], ...] = (
    ("cabernet-sauvignon", None),
    ("pinot-noir", None),
    ("chardonnay", 5),
    ("riesling", 5),
    ("nebbiolo", None),
    ("tempranillo", None),
    ("syrah/shiraz", None),  # X-Wines labels this grape "Syrah/Shiraz"
    ("sangiovese", None),
)
RESULTS_TABLE_TOP_N = 10  # rows per ranking table in RESULTS.md
RESULTS_OUTLIER_TABLE_N = 20  # §4.2 shows more rows — the shortlist is the point
RESULTS_QUANTILE_NOMINAL = 0.8  # 0.1/0.9 heads → nominal 80% band, the coverage target
# §4.4 value table: best-rated monogrape wines priced under this euro cap. Only
# renders when the price snapshot exists; degrades to a note otherwise.
RESULTS_VALUE_CAP_EUR = 50.0
RESULTS_VALUE_TABLE_N = 15

# Classic blends the monogrape default hides, featured region-filtered in §3.3:
# (RegionName matched case-insensitively, display heading). Amarone della
# Valpolicella is a Corvina-based blend, so it never reaches the grape tables.
RESULTS_SHOWCASE_REGIONS: tuple[tuple[str, str], ...] = (
    ("amarone della valpolicella", "Amarone della Valpolicella"),
    ("amarone della valpolicella classico", "Amarone della Valpolicella Classico"),
)

# §8 qualitative sanity check: (name substring for eval/sanity.py, preferred
# vintage). Substrings avoid accents so a cp1252 console never trips; a null
# vintage falls back to each wine's most-rated one.
RESULTS_SANITY_QUERIES: tuple[tuple[str, int | None], ...] = (
    ("masi costasera", 2016),  # Amarone della Valpolicella
    ("marchesi di barolo", None),  # Nebbiolo
    ("sessantanni", None),  # Primitivo di Manduria
    ("nero d'avola", None),  # Nero d'Avola
    ("tignanello", None),  # Super Tuscan
    ("brunello", None),  # Sangiovese
    ("de riscal", None),  # Tempranillo (Marques de Riscal; "riscal" alone hits "Friscale")
    ("beaujolais", None),  # Beaujolais (Gamay)
    ("neuf-du-pape", None),  # Southern-Rhone GSM
    ("guigal", None),  # Cotes du Rhone
)


def _project_root() -> Path:
    """Walk up from this file until we find the repo's `pyproject.toml`.

    Anchoring relative paths to the repo root (instead of CWD) is what lets
    notebooks, scripts, and tests resolve `data/raw/...` the same way no
    matter where they were launched from.
    """
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd()


class Settings(BaseSettings):
    """Environment-driven configuration with sensible local-dev defaults."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="VININATOR_",
        extra="ignore",
    )

    data_dir: Path = Field(default=Path("data"))
    xwines_variant: Literal["test", "slim", "full"] = Field(default="test")
    nominatim_contact: str = Field(default="")

    @field_validator("data_dir", mode="after")
    @classmethod
    def _anchor_to_project_root(cls, value: Path) -> Path:
        """Relative `data_dir` resolves against the repo root, not CWD."""
        return value if value.is_absolute() else (_project_root() / value).resolve()

    @computed_field  # type: ignore[prop-decorator]
    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def interim_dir(self) -> Path:
        return self.data_dir / "interim"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def xwines_wines_parquet(self) -> Path:
        """Per-wine canonical parquet (WineID + attrs)."""
        return self.raw_dir / XWINES_WINES_PARQUET

    @computed_field  # type: ignore[prop-decorator]
    @property
    def xwines_ratings_parquet(self) -> Path:
        """Per-rating canonical parquet (RatingID, WineID, Vintage, Rating, Date)."""
        return self.raw_dir / XWINES_RATINGS_PARQUET

    @computed_field  # type: ignore[prop-decorator]
    @property
    def xwines_wines_csv(self) -> Path:
        """Source CSV the variant expects to find under `data/raw/`."""
        return self.raw_dir / XWINES_VARIANTS[self.xwines_variant]["wines_csv"]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def xwines_ratings_csv(self) -> Path:
        return self.raw_dir / XWINES_VARIANTS[self.xwines_variant]["ratings_csv"]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def geocode_parquet(self) -> Path:
        """Cache of `RegionName` → `(lat, lon)` results from Nominatim."""
        return self.interim_dir / GEOCODE_PARQUET

    @computed_field  # type: ignore[prop-decorator]
    @property
    def nominatim_user_agent(self) -> str:
        """User agent sent to Nominatim, built from the env-driven contact."""
        if not self.nominatim_contact:
            raise RuntimeError(
                "VININATOR_NOMINATIM_CONTACT is unset. Nominatim's TOS requires a "
                "contactable operator string (e.g. '+https://github.com/<you>/<fork>' "
                "or 'mailto:you@example.com'). Set it in your local .env before "
                "running geocoding."
            )
        return NOMINATIM_USER_AGENT_TEMPLATE.format(contact=self.nominatim_contact)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def soil_raw_dir(self) -> Path:
        """One JSON per region: the raw SoilGrids 3x3 buffer response."""
        return self.interim_dir / SOIL_RAW_DIRNAME

    @computed_field  # type: ignore[prop-decorator]
    @property
    def dem_raw_dir(self) -> Path:
        """One JSON per region: the raw Open-Elevation 3x3 grid response."""
        return self.interim_dir / DEM_RAW_DIRNAME

    @computed_field  # type: ignore[prop-decorator]
    @property
    def soil_parquet(self) -> Path:
        """Aggregated per-region soil + terrain features."""
        return self.interim_dir / SOIL_PARQUET

    @computed_field  # type: ignore[prop-decorator]
    @property
    def nasa_power_raw_dir(self) -> Path:
        """One JSON per region: the raw NASA POWER Daily API response."""
        return self.raw_dir / NASA_POWER_RAW_DIRNAME

    @computed_field  # type: ignore[prop-decorator]
    @property
    def climate_parquet(self) -> Path:
        """Per (region, vintage_year) climate features + anomalies."""
        return self.interim_dir / CLIMATE_PARQUET

    @computed_field  # type: ignore[prop-decorator]
    @property
    def climatology_parquet(self) -> Path:
        """Per-region long-form climatology means over CLIMATOLOGY_WINDOW."""
        return self.interim_dir / CLIMATOLOGY_PARQUET

    @computed_field  # type: ignore[prop-decorator]
    @property
    def terroir_parquet(self) -> Path:
        """climate.parquet ⨝ soil.parquet, keyed on (region, country, vintage_year)."""
        return self.interim_dir / TERROIR_PARQUET

    @computed_field  # type: ignore[prop-decorator]
    @property
    def processed_train_parquet(self) -> Path:
        """Final training table (one row per rating, vintage ≤ 2018, train WineIDs)."""
        return self.processed_dir / PROCESSED_TRAIN_PARQUET

    @computed_field  # type: ignore[prop-decorator]
    @property
    def processed_test_parquet(self) -> Path:
        """Held-out wine test set (one row per rating, vintage ≤ 2018, test WineIDs)."""
        return self.processed_dir / PROCESSED_TEST_PARQUET

    @computed_field  # type: ignore[prop-decorator]
    @property
    def processed_future_vintage_test_parquet(self) -> Path:
        """Future-vintage holdout (vintage 2019–2021, any WineID)."""
        return self.processed_dir / PROCESSED_FUTURE_VINTAGE_TEST_PARQUET

    @computed_field  # type: ignore[prop-decorator]
    @property
    def ablations_parquet(self) -> Path:
        """Phase 5 ablation grid (arm × split × metrics), one row per pair."""
        return self.processed_dir / ABLATIONS_PARQUET

    @computed_field  # type: ignore[prop-decorator]
    @property
    def shap_importance_parquet(self) -> Path:
        """Phase 5 mean-|SHAP| per feature on the rating model."""
        return self.processed_dir / SHAP_IMPORTANCE_PARQUET

    @computed_field  # type: ignore[prop-decorator]
    @property
    def recommendations_drink_now_parquet(self) -> Path:
        """Phase 6 drink-now ranking (top wines at one opening year)."""
        return self.processed_dir / RECOMMENDATIONS_DRINK_NOW_PARQUET

    @computed_field  # type: ignore[prop-decorator]
    @property
    def recommendations_age_well_parquet(self) -> Path:
        """Phase 6 age-well long-format sweep (one row per wine × opening year)."""
        return self.processed_dir / RECOMMENDATIONS_AGE_WELL_PARQUET

    @computed_field  # type: ignore[prop-decorator]
    @property
    def recommendations_age_well_summary_parquet(self) -> Path:
        """Phase 6 age-well per-wine summary (peak year/rating, slope, trajectory)."""
        return self.processed_dir / RECOMMENDATIONS_AGE_WELL_SUMMARY_PARQUET

    @computed_field  # type: ignore[prop-decorator]
    @property
    def recommendations_standout_years_parquet(self) -> Path:
        """Phase 6 standout-of-the-year shortlists (one block per opening year)."""
        return self.processed_dir / RECOMMENDATIONS_STANDOUT_YEARS_PARQUET

    @computed_field  # type: ignore[prop-decorator]
    @property
    def recommendations_outliers_parquet(self) -> Path:
        """Phase 6 overperformer outliers (predicted rating vs. peer baseline)."""
        return self.processed_dir / RECOMMENDATIONS_OUTLIERS_PARQUET

    @computed_field  # type: ignore[prop-decorator]
    @property
    def wine_reviews_csv(self) -> Path:
        """Kaggle Wine Reviews price snapshot (manual drop, immutable)."""
        return self.raw_dir / WINE_REVIEWS_DIRNAME / WINE_REVIEWS_CSV

    @computed_field  # type: ignore[prop-decorator]
    @property
    def price_parquet(self) -> Path:
        """Per-wine price estimates matched from the Wine Reviews snapshot."""
        return self.interim_dir / PRICE_PARQUET

    @computed_field  # type: ignore[prop-decorator]
    @property
    def recommendations_library_dir(self) -> Path:
        """The browsable per-grape markdown library (scripts/build_library.py)."""
        return _project_root() / "reports" / "recommendations"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def library_tables_dir(self) -> Path:
        """Per-section ranking parquets backing the library pages. Gitignored."""
        return self.recommendations_library_dir / "tables"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def results_md(self) -> Path:
        """The generated RESULTS.md at the repo root (Phase 7 deliverable)."""
        return _project_root() / "RESULTS.md"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def report_tables_dir(self) -> Path:
        """Per-grape ranking parquets backing the RESULTS.md §3 tables."""
        return _project_root() / "reports" / "tables"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def figures_dir(self) -> Path:
        """Figures embedded by RESULTS.md (SHAP plots, ablation charts)."""
        return _project_root() / "reports" / "figures"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def models_dir(self) -> Path:
        """Trained model bundles (`.cbm` + `.meta.json`). Gitignored via data/."""
        return self.data_dir / MODELS_DIRNAME

    @computed_field  # type: ignore[prop-decorator]
    @property
    def snapshots_dir(self) -> Path:
        """CatBoost training snapshots for crash recovery. Deleted on successful save."""
        return self.data_dir / "snapshots"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def mlflow_tracking_dir(self) -> Path:
        """Local MLflow file store at the repo root (gitignored as `mlruns/`)."""
        return _project_root() / MLFLOW_RUNS_DIRNAME

    @computed_field  # type: ignore[prop-decorator]
    @property
    def mlflow_tracking_uri(self) -> str:
        """`file://` URI MLflow expects for a local store."""
        return self.mlflow_tracking_dir.as_uri()

    def ensure_dirs(self) -> None:
        """Create the data layout if missing. Idempotent."""
        for d in (
            self.raw_dir,
            self.interim_dir,
            self.processed_dir,
            self.soil_raw_dir,
            self.dem_raw_dir,
            self.nasa_power_raw_dir,
            self.models_dir,
            self.snapshots_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()
