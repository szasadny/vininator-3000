"""Phase 4 modeling package: rating, profile, and harmonize CatBoost models.

`dataset.py` owns the feature contract (which columns are features vs targets,
which are categorical). `artifacts.py` and `tracking.py` are shared plumbing.
The three trainer modules (`rating`, `profile`, `harmonize`) consume the
contract and never re-derive it.
"""

from __future__ import annotations
