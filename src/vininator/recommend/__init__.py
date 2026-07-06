"""Phase 6 recommender: turn the rating model into rankings.

Every ranking is the trained rating model scored at a chosen `age_at_review`
(`opening_year − vintage_year`). Vintage — and therefore the terroir block — is
held constant, so there is no live NASA POWER / SoilGrids fetch and no feature
re-engineering: the recommender only loads saved bundles and the already-built
processed parquets, exactly like `eval/sanity.py`.

`drink_now.py` owns the candidate table and the shared score/enrich path;
`age_well.py`, `standout_years.py`, and `outliers.py` reuse it rather than
re-implementing scoring, so a wine's rank is consistent across every ranking.
"""

from __future__ import annotations
