"""Generate the browsable recommendation library under reports/recommendations/.

Loops the Phase 6 recommender over every well-covered grape (recent vintages,
monogrape, grouped by wine type), plus expanded lists for the house favorites and
a best-years guide. Each catalog entry carries post-hoc price/value metadata from
the Wine Reviews snapshot — never a model feature, joined after scoring.

Run after the models exist and (optionally) after `vininator features price`:

    PYTHONIOENCODING=utf-8 uv run python scripts/build_library.py

Without the price snapshot the pages still generate; the value tables show a
"no price source" note instead.
"""

from __future__ import annotations

import gc
from collections.abc import Callable

import polars as pl

from vininator.config import (
    LIBRARY_FAVORITE_REGIONS,
    LIBRARY_FAVORITE_TOP_N,
    LIBRARY_FUTURE_PEAK_MIN_YEAR,
    LIBRARY_MIN_VINTAGE,
    LIBRARY_TOP_N,
    LIBRARY_VINTAGE_QUALITY_AGE,
    LIBRARY_WINE_TYPES,
    STANDOUT_YEAR_RANGE,
    VALUE_PRICE_CAP_EUR,
    WINE_REVIEWS_URL,
    get_settings,
)
from vininator.eval.report_data import (
    age_well_display,
    drink_now_display,
    markdown_table,
    value_display,
    write_markdown_atomic,
)
from vininator.recommend.drink_now import build_candidates, load_recommend_bundles
from vininator.recommend.library import (
    CatalogSection,
    build_catalog_section,
    favorite_slices,
    filter_library_candidates,
    filter_vintage_window,
    join_price,
    list_grape_groups,
    vintage_quality,
)
from vininator.recommend.standout_years import recommend_standout_years

_STYLE_LABELS = frozenset(display for display, _ in LIBRARY_FAVORITE_REGIONS)
_TYPE_FILENAME = {"Red": "red.md", "White": "white.md", "Rosé": "rose.md"}


def _note(message: str) -> None:
    print(message, flush=True)


# ---------------------------------------------------------------------------
# Section rendering
# ---------------------------------------------------------------------------


def _library_drink_now_display(table: pl.DataFrame) -> pl.DataFrame:
    """Drink-now display with a price column appended (library pages only)."""
    base = drink_now_display(table)
    price_eur = table.get_column("price_eur").to_list()
    return base.with_columns(
        pl.Series("Price (EUR)", [f"€{p:.0f}" if p is not None else "" for p in price_eur]),
        pl.Series("Band", table.get_column("price_band").to_list()),
    )


def _table_or_note(
    df: pl.DataFrame, display: Callable[[pl.DataFrame], pl.DataFrame], note: str
) -> str:
    """Render `df` via `display`, or a single italic note line when empty."""
    return markdown_table(display(df)) if not df.is_empty() else note


def _section_md(section: CatalogSection) -> str:
    """One grape/style block: coverage line + the five ranking tables."""
    pct = 100.0 * section.n_priced / section.n_bottles if section.n_bottles else 0.0
    heading = f"### {section.label} ({section.n_bottles} bottles, {pct:.0f}% exact-priced)"
    style_note = (
        "\n_Wine style: a region-defined blend, not a single grape._\n"
        if section.label in _STYLE_LABELS
        else ""
    )
    cap = int(VALUE_PRICE_CAP_EUR)
    coverage_caveat = (
        "Exact-priced bottles only (winery-median estimates underprice flagships, so they are "
        "excluded from value); cheap or obscure wines may be missing, not bad."
    )
    return "\n".join(
        [
            heading,
            style_note,
            "**Drink now (2026)** — top by predicted rating.",
            "",
            _table_or_note(section.drink_now, _library_drink_now_display, "_No candidates._"),
            "",
            "**Age well (2026–2036)** — top rising or late-peaking bottles by peak rating.",
            "",
            _table_or_note(section.age_well, age_well_display, "_No cellar candidates._"),
            "",
            f"**Future greats (peak ≥ {LIBRARY_FUTURE_PEAK_MIN_YEAR})** — young bottles predicted "
            "to peak years from now.",
            "",
            _table_or_note(
                section.future_greats, age_well_display, "_No future-great candidates._"
            ),
            "",
            f"**Best value (under €{cap})** — best-rated priced bottles at or under €{cap}. "
            + coverage_caveat,
            "",
            _table_or_note(section.best_value, value_display, "_No priced bottles under the cap._"),
            "",
            "**Best value (rating per €)** — highest predicted rating per euro. " + coverage_caveat,
            "",
            _table_or_note(section.best_value_score, value_display, "_No priced bottles._"),
            "",
        ]
    )


# ---------------------------------------------------------------------------
# Page builders
# ---------------------------------------------------------------------------


def _index_md(type_counts: dict[str, int], coverage: dict[str, int], has_price: bool) -> str:
    total = coverage["total"]
    if has_price and total:
        pct = 100.0 * coverage["priced"] / total
        pct_exact = 100.0 * coverage["exact"] / total
        pct_winery = 100.0 * coverage["winery"] / total
        price_line = (
            f"Prices are matched for {pct:.0f}% of catalogued bottles ({pct_exact:.0f}% exact, "
            f"{pct_winery:.0f}% winery-median). Value tables use the **exact** matches only — the "
            "winery-median estimate underprices a winery's flagship — so their coverage is the "
            "exact share. Coverage skews to famous and expensive wines."
        )
    else:
        price_line = (
            "No price snapshot was found, so value tables are empty. Add one with "
            "`uv run vininator features price` (see the pipeline in the root README)."
        )
    type_links = "\n".join(
        f"- **[{t}]({_TYPE_FILENAME[t]})** — {type_counts.get(t, 0)} grape sections"
        for t in LIBRARY_WINE_TYPES
    )
    return "\n".join(
        [
            "# Vininator 3000 — Recommendation library",
            "",
            f"Per-grape and per-favorite rankings for recent wines (vintages "
            f"{LIBRARY_MIN_VINTAGE + 1}–2021), scored at opening year 2026. Drink-now, age-well, "
            "future greats, and best-value picks. Monogrape wines only on the type pages; the "
            "favorites include region-defined blends. Model quality and caveats live in "
            "[RESULTS.md](../../RESULTS.md).",
            "",
            price_line,
            "",
            "## Pages",
            "",
            "- **[Favorites](favorites.md)** — the house picks, deeper lists.",
            "- **[Best years](best-years.md)** — standout drinking years and best vintages.",
            type_links,
            "",
            "## Price data",
            "",
            "Prices come from the Kaggle [Wine Reviews](" + WINE_REVIEWS_URL + ") dataset "
            "(zynicide, 2017 snapshot, USD), matched to X-Wines by winery and wine name and shown "
            "in EUR. **License: CC BY-NC-SA 4.0 — non-commercial.** Do not reuse the value tables "
            "commercially.",
            "",
        ]
    )


def _type_page_md(wine_type: str, sections: list[CatalogSection]) -> str:
    intro = (
        f"# {wine_type} wines\n\n"
        f"Monogrape {wine_type.lower()} wines by grape, biggest sections first. "
        "See the [library index](README.md) for how these are built."
    )
    return intro + "\n\n" + "\n".join(_section_md(s) for s in sections)


def _favorites_md(sections: list[CatalogSection]) -> str:
    intro = (
        "# Favorite wines\n\n"
        "Deeper lists for the house favorites — three grapes (Sangiovese, Primitivo, "
        "Nero d'Avola) and two styles (Amarone della Valpolicella, Super Tuscan). Styles are "
        "region-defined blends, so they are filtered by region, not grape."
    )
    return intro + "\n\n" + "\n".join(_section_md(s) for s in sections)


def _best_years_md(standout, vintage_q: pl.DataFrame) -> str:
    parts = [
        "# Best years",
        "",
        "## Best drinking years (2026–2031)",
        "",
        "One shortlist per year — the wines predicted to be at their best when opened that year. "
        "A wine can recur when its trajectory plateaus.",
        "",
    ]
    if standout is None:
        parts.append("_No standout shortlists (no candidates matched)._")
    else:
        for year in standout.years_covered:
            block = standout.table.filter(pl.col("opening_year") == year)
            parts.append(f"### {year}")
            parts.append("")
            parts.append(_table_or_note(block, drink_now_display, "_No candidates._"))
            parts.append("")

    parts += [
        "## Best vintages",
        "",
        f"Vintages ranked by mean predicted rating, each scored at the same age "
        f"({LIBRARY_VINTAGE_QUALITY_AGE} years post-vintage) so the age effect doesn't confound "
        "vintage quality. Monogrape wines only.",
        "",
        _table_or_note(vintage_q, _vintage_quality_display, "_No vintages in range._"),
        "",
    ]
    return "\n".join(parts)


def _vintage_quality_display(table: pl.DataFrame) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "Type": table.get_column("wine_type").to_list(),
            "Vintage": table.get_column("vintage_year").to_list(),
            "Bottles": table.get_column("n_bottles").to_list(),
            "Mean predicted rating": [
                f"{x:.3f}" for x in table.get_column("mean_predicted_rating").to_list()
            ],
        }
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def main() -> None:
    settings = get_settings()
    settings.library_tables_dir.mkdir(parents=True, exist_ok=True)

    _note("[1/5] loading bundles + candidates")
    bundles = load_recommend_bundles()
    window = filter_vintage_window(build_candidates(_note))
    gc.collect()

    price = _load_price(settings)
    window = join_price(window, price)
    general = filter_library_candidates(window)
    tables_dir = settings.library_tables_dir

    _note("[2/5] per-grape sections")
    groups = list_grape_groups(general)
    by_type: dict[str, list[CatalogSection]] = {t: [] for t in LIBRARY_WINE_TYPES}
    for row in groups.iter_rows(named=True):
        wine_type, grape = row["wine_type"], row["grape_majority"]
        section_slice = general.filter(
            (pl.col("wine_type") == wine_type) & (pl.col("grape_majority") == grape)
        )
        section = build_catalog_section(
            wine_type, grape, section_slice, bundles, tables_dir, top=LIBRARY_TOP_N, notify_fn=_note
        )
        by_type[wine_type].append(section)

    _note("[3/5] favorites")
    favorites = [
        build_catalog_section(
            "Favorite",
            label,
            slice_,
            bundles,
            tables_dir,
            top=LIBRARY_FAVORITE_TOP_N,
            notify_fn=_note,
        )
        for label, slice_ in favorite_slices(window)
    ]

    _note("[4/5] best years")
    standout = _standout(general, bundles, tables_dir)
    vintage_q = vintage_quality(general, bundles)

    _note("[5/5] rendering pages")
    coverage = _coverage(general)
    type_counts = {t: len(by_type[t]) for t in LIBRARY_WINE_TYPES}
    _write_coverage_parquet(general, tables_dir)

    lib = settings.recommendations_library_dir
    write_markdown_atomic(lib / "README.md", _index_md(type_counts, coverage, price is not None))
    write_markdown_atomic(lib / "favorites.md", _favorites_md(favorites))
    write_markdown_atomic(lib / "best-years.md", _best_years_md(standout, vintage_q))
    for wine_type in LIBRARY_WINE_TYPES:
        write_markdown_atomic(
            lib / _TYPE_FILENAME[wine_type], _type_page_md(wine_type, by_type[wine_type])
        )

    _note(f"wrote {lib / 'README.md'} + favorites/best-years + {len(_TYPE_FILENAME)} type pages")
    _report_coverage(general)


def _standout(general: pl.DataFrame, bundles, tables_dir):
    """Standout-of-the-year report, or None when no candidates match any year."""
    from vininator.recommend.drink_now import RecommendFilters

    try:
        return recommend_standout_years(
            from_year=STANDOUT_YEAR_RANGE[0],
            to_year=STANDOUT_YEAR_RANGE[1],
            filters=RecommendFilters(monogrape=False),
            top=LIBRARY_TOP_N,
            distinct_wines=True,
            candidates=general,
            bundles=bundles,
            out_path=tables_dir / "standout_years.parquet",
        )
    except ValueError:
        return None


def _load_price(settings) -> pl.DataFrame | None:
    path = settings.price_parquet
    if not path.exists():
        _note(
            f"    no price snapshot at {path} — value tables will be empty. "
            "Run `uv run vininator features price` to add prices."
        )
        return None
    _note(f"    joined price snapshot from {path}")
    return pl.read_parquet(path)


def _coverage(general: pl.DataFrame) -> dict[str, int]:
    return {
        "total": general.height,
        "priced": general.filter(pl.col("price_estimate").is_not_null()).height,
        "exact": general.filter(pl.col("match_confidence") == "exact").height,
        "winery": general.filter(pl.col("match_confidence") == "winery-median").height,
    }


def _write_coverage_parquet(general: pl.DataFrame, tables_dir) -> None:
    cov = (
        general.group_by(["wine_type", "grape_majority"])
        .agg(
            pl.len().alias("n_bottles"),
            pl.col("price_estimate").is_not_null().sum().alias("n_priced"),
            (pl.col("match_confidence") == "exact").sum().alias("n_exact"),
        )
        .sort(["wine_type", "n_bottles"], descending=[False, True])
    )
    cov.write_parquet(tables_dir / "price_coverage.parquet")


def _report_coverage(general: pl.DataFrame) -> None:
    cov = _coverage(general)
    total = cov["total"] or 1
    _note(
        f"price coverage: {cov['priced']}/{cov['total']} bottles "
        f"({100.0 * cov['priced'] / total:.1f}%); "
        f"exact {cov['exact']}, winery-median {cov['winery']}"
    )


if __name__ == "__main__":
    main()
