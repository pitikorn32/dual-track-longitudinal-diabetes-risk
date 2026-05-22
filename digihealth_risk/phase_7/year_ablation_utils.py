"""Phase 7 year-features ablation — shared helpers.

The ablation drops three calendar-time features from training so the model
becomes invariant to the calendar year a patient is scored in:

    Year, Year_centered, Year_centered_sq

Patient-relative time signals (`Age`, `years_since_last_fbs`, `has_fbs_this_year`,
`is_missing_last_year`, history-window slope/range stats) are retained.

The final v2 workflow uses one canonical tree-training helper:

  * `digihealth_risk.phase_2.train_tree_models`

`patch_drop_year_features()` patches that module so the ablation applies
uniformly to every downstream training entry point.

Call `patch_drop_year_features()` at the very top of each phase_7 script,
before importing any phase_2 / phase_4 / phase_5 helpers.
"""

from __future__ import annotations

import importlib
from types import ModuleType

import pandas as pd

YEAR_FEATURES: tuple[str, ...] = ("Year_centered", "Year_centered_sq")
RAW_YEAR_FEATURE: str = "Year"

_TARGET_MODULES: tuple[str, ...] = ("digihealth_risk.phase_2.train_tree_models",)

_PATCHED = False


def _patch_one(module: ModuleType) -> None:
    module.LEAKAGE_OR_METADATA_COLUMNS = set(module.LEAKAGE_OR_METADATA_COLUMNS) | {
        RAW_YEAR_FEATURE,
        *YEAR_FEATURES,
    }
    original_engineer = module.engineer_features

    def engineer_features_no_year(df: pd.DataFrame) -> pd.DataFrame:
        out = original_engineer(df)
        drop_cols = [c for c in YEAR_FEATURES if c in out.columns]
        if drop_cols:
            out = out.drop(columns=drop_cols)
        return out

    module.engineer_features = engineer_features_no_year


def patch_drop_year_features() -> None:
    """Monkey-patch the phase_2 v2 module to exclude Year features.

    Idempotent — safe to call from multiple entry points in the same process.
    """
    global _PATCHED
    if _PATCHED:
        return
    for name in _TARGET_MODULES:
        module = importlib.import_module(name)
        _patch_one(module)
    _PATCHED = True


def dropped_feature_names() -> tuple[str, ...]:
    return (RAW_YEAR_FEATURE, *YEAR_FEATURES)
