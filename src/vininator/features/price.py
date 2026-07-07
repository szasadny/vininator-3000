"""Post-hoc price metadata for the recommendation library.

Price is NOT a model feature — it never enters a CatBoost pool, and no model is
retrained here. This module only sources a static price snapshot and matches it
to X-Wines wines, so the library can decorate its already-scored rankings with a
price and a value view. The join happens downstream in `recommend/library.py`.

Source: the Kaggle "Wine Reviews" dataset (zynicide), a 2017 snapshot of Wine
Enthusiast reviews with a USD `price` per bottle. It is a static CSV — manual
drop under `data/raw/wine_reviews/`, like the slim/full X-Wines variants — so
there is no auth-gated fetch loop; the derived `price.parquet` IS the cache.

Matching is the hard part: X-Wines and Wine Reviews share no key, so we match on
normalized winery + wine-name tokens, deterministically and without a fuzzy
library:

- **exact** — same normalized winery AND every token of the X-Wines wine name
  appears in the Wine-Reviews title. Price = median of the matching source rows.
- **winery-median** — same normalized winery with at least `PRICE_MIN_WINERY_ROWS`
  priced source rows. Price = the winery's median. (The spec floated a
  winery+region median; Kaggle's `region_1`/`province` almost never equals the
  X-Wines `RegionName`, so requiring region equality would gut the match rate.
  We drop region from the key and note the coarsening here.)
- **none** — no match; the wine gets no price row (the join fills null later).

Vintage is deliberately ignored: the 2017 snapshot has at most a handful of
vintages per wine and no reliable per-vintage price, so a single wine-level
estimate is the honest resolution. Coverage skews to the famous/expensive tail
(those wines get reviewed); cheap and obscure wines are mostly `none`, which is
why the library always prints a per-grape coverage line and never presents a
value ranking as if every wine were priced.

⚠️ License: the Wine Reviews dataset is CC BY-NC-SA 4.0 (non-commercial). Any
derived table published from it must carry that attribution.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable

import polars as pl

from vininator.config import (
    PRICE_MIN_WINERY_ROWS,
    PRICE_SOURCE,
    PRICE_SOURCE_CURRENCY,
    get_settings,
)

NotifyFn = Callable[[str], None]

PRICE_SCHEMA: dict[str, pl.DataType] = {
    "wine_id": pl.Int64(),
    "price_estimate": pl.Float64(),  # source-native (USD); null never stored here
    "price_source": pl.String(),
    "price_currency": pl.String(),
    "match_confidence": pl.String(),  # "exact" | "winery-median"
}


# ---------------------------------------------------------------------------
# Pure normalization
# ---------------------------------------------------------------------------


def _norm(value: str | None) -> str:
    """Fold to lowercase ASCII and collapse punctuation/whitespace to spaces.

    The one matching primitive: `"Château Léoville-Barton"` and
    `"Chateau Leoville Barton"` normalize equal so accents and hyphens don't
    break the winery join.
    """
    if value is None:
        return ""
    folded = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"[^a-z0-9]+", " ", folded).strip()


def _tokens(value: str | None) -> list[str]:
    """Unique, order-preserving normalized tokens of `value` (drops empties)."""
    return list(dict.fromkeys(t for t in _norm(value).split(" ") if t))


# ---------------------------------------------------------------------------
# Source loading
# ---------------------------------------------------------------------------


def load_price_source() -> pl.DataFrame:
    """Read the Wine Reviews CSV → `norm_winery`, `title_tokens`, `price` (USD).

    Raises `FileNotFoundError` (never silently skips) when the manual-drop CSV is
    absent, pointing at the Kaggle page and the exact path it must live at.
    """
    settings = get_settings()
    csv_path = settings.wine_reviews_csv
    if not csv_path.exists():
        from vininator.config import WINE_REVIEWS_URL

        raise FileNotFoundError(
            f"Wine Reviews price snapshot not found at {csv_path}.\n"
            f"Download it from {WINE_REVIEWS_URL} (file 'winemag-data-130k-v2.csv', "
            "CC BY-NC-SA 4.0 non-commercial) and drop it there, then re-run "
            "`uv run vininator features price`."
        )

    raw = pl.read_csv(csv_path, columns=["winery", "title", "price"], infer_schema_length=None)
    return (
        raw.with_columns(pl.col("price").cast(pl.Float64, strict=False))
        .filter(pl.col("price").is_not_null() & (pl.col("price") > 0))
        .with_columns(
            pl.col("winery").map_elements(_norm, return_dtype=pl.String).alias("norm_winery"),
            pl.col("title")
            .map_elements(_tokens, return_dtype=pl.List(pl.String))
            .alias("title_tokens"),
        )
        .filter(pl.col("norm_winery").str.len_chars() > 0)
        .select("norm_winery", "title_tokens", "price")
    )


def load_xwines_for_match() -> pl.DataFrame:
    """X-Wines wines → `wine_id`, `norm_winery`, `wine_tokens` for matching."""
    settings = get_settings()
    return (
        pl.scan_parquet(settings.xwines_wines_parquet)
        .select(
            pl.col("WineID").alias("wine_id"),
            pl.col("WineryName").alias("winery_name"),
            pl.col("WineName").alias("wine_name"),
        )
        .collect()
        .with_columns(
            pl.col("winery_name").map_elements(_norm, return_dtype=pl.String).alias("norm_winery"),
            pl.col("wine_name")
            .map_elements(_tokens, return_dtype=pl.List(pl.String))
            .alias("wine_tokens"),
        )
        .filter(pl.col("norm_winery").str.len_chars() > 0)
    )


# ---------------------------------------------------------------------------
# Matching (pure over the two normalized frames)
# ---------------------------------------------------------------------------


def match_prices(wines: pl.DataFrame, source: pl.DataFrame) -> pl.DataFrame:
    """Tiered winery/name match → one priced row per matched `wine_id`.

    Deterministic: exact (name-token containment within a winery) wins over the
    winery median; both reduce many source rows to a median, so there is no RNG
    and no order dependence. Returns only matched wines — unmatched ones are
    absent (the downstream join fills them null), which keeps the parquet small.
    """
    exact = _match_exact(wines, source)
    winery_med = _match_winery_median(wines, source)

    priced = (
        wines.select("wine_id")
        .join(exact, on="wine_id", how="left")
        .join(winery_med, on="wine_id", how="left")
        .with_columns(
            pl.coalesce("price_exact", "price_winery").alias("price_estimate"),
            pl.when(pl.col("price_exact").is_not_null())
            .then(pl.lit("exact"))
            .when(pl.col("price_winery").is_not_null())
            .then(pl.lit("winery-median"))
            .otherwise(pl.lit("none"))
            .alias("match_confidence"),
        )
        .filter(pl.col("match_confidence") != "none")
        .with_columns(
            pl.lit(PRICE_SOURCE).alias("price_source"),
            pl.lit(PRICE_SOURCE_CURRENCY).alias("price_currency"),
        )
        .select(list(PRICE_SCHEMA.keys()))
    )
    return priced.cast(PRICE_SCHEMA)  # type: ignore[arg-type]


def _match_exact(wines: pl.DataFrame, source: pl.DataFrame) -> pl.DataFrame:
    """`wine_id` → median price of source rows whose title contains every name token."""
    named = wines.filter(pl.col("wine_tokens").list.len() >= 1)
    pairs = named.join(source, on="norm_winery", how="inner")
    if pairs.is_empty():
        return pl.DataFrame(schema={"wine_id": pl.Int64(), "price_exact": pl.Float64()})
    return (
        pairs.filter(
            pl.col("wine_tokens").list.set_intersection(pl.col("title_tokens")).list.len()
            == pl.col("wine_tokens").list.len()
        )
        .group_by("wine_id")
        .agg(pl.col("price").median().alias("price_exact"))
    )


def _match_winery_median(wines: pl.DataFrame, source: pl.DataFrame) -> pl.DataFrame:
    """`wine_id` → the winery's median price, when the winery has enough rows."""
    winery_med = (
        source.group_by("norm_winery")
        .agg(pl.col("price").median().alias("price_winery"), pl.len().alias("n_rows"))
        .filter(pl.col("n_rows") >= PRICE_MIN_WINERY_ROWS)
        .select("norm_winery", "price_winery")
    )
    return wines.join(winery_med, on="norm_winery", how="inner").select("wine_id", "price_winery")


# ---------------------------------------------------------------------------
# Build orchestrator
# ---------------------------------------------------------------------------


def build_price_table(force: bool = False, notify_fn: NotifyFn | None = None) -> pl.DataFrame:
    """Build `data/interim/price.parquet` from the Wine Reviews snapshot.

    Cache-first: returns the existing parquet untouched unless `force`. Writes
    atomically (tmp → rename). Returns the priced frame; logs per-tier counts and
    the overall match rate against the X-Wines catalog via `notify_fn`.
    """
    settings = get_settings()
    settings.ensure_dirs()
    out_path = settings.price_parquet

    if out_path.exists() and not force:
        _notify(notify_fn, f"... price cache present at {out_path} (use force=True to rebuild)")
        return pl.read_parquet(out_path)

    source = load_price_source()
    wines = load_xwines_for_match()
    priced = match_prices(wines, source)

    _write_parquet_atomic(priced, out_path)

    n_total = wines.height
    counts = dict(
        priced.group_by("match_confidence").len().iter_rows()  # {conf: n}
    )
    n_exact = counts.get("exact", 0)
    n_winery = counts.get("winery-median", 0)
    n_priced = n_exact + n_winery
    pct = 100.0 * n_priced / n_total if n_total else 0.0
    _notify(notify_fn, f"... source rows (priced): {source.height:,}")
    _notify(notify_fn, f"... exact matches:        {n_exact:,}")
    _notify(notify_fn, f"... winery-median:        {n_winery:,}")
    _notify(notify_fn, f"... unpriced:             {n_total - n_priced:,}")
    _notify(notify_fn, f"... coverage:             {pct:.1f}% of {n_total:,} wines")
    return priced


def scan_price() -> pl.LazyFrame:
    """Lazy frame over the price cache. Raises if it hasn't been built."""
    path = get_settings().price_parquet
    if not path.exists():
        raise FileNotFoundError(
            f"Price parquet not found at {path}. Run `uv run vininator features price` first."
        )
    return pl.scan_parquet(path)


def _notify(notify_fn: NotifyFn | None, message: str) -> None:
    if notify_fn is not None:
        notify_fn(message)


def _write_parquet_atomic(df: pl.DataFrame, target) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    df.write_parquet(tmp)
    tmp.replace(target)
