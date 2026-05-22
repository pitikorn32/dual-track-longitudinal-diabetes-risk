"""Canonical patient-level train/calibration/test split.

Every phase script should import `apply_canonical_split` from here so cross-family
comparison runs on the same test patients regardless of which modeling table the
phase loads.

The split is derived from `datasets/df_final.pkl` (the source of truth for the
6,892-patient cohort) using:
    - `RANDOM_SEED = 20260501`
    - `TEST_PATIENT_FRACTION = 0.20`
    - `CALIBRATION_PATIENT_FRACTION = 0.20`  (of total, not of remainder)
    - patient ids cast to str and sorted before sampling

The split is cached at `digihealth_risk/phase_0/outputs/patient_split.csv`. The
helper rebuilds the cache if the file is missing.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

RANDOM_SEED = 20260501
TEST_PATIENT_FRACTION = 0.20
CALIBRATION_PATIENT_FRACTION = 0.20

ROOT = Path(__file__).resolve().parents[2]
SOURCE_DATA = ROOT / "datasets" / "df_final.pkl"
SPLIT_CACHE = ROOT / "digihealth_risk" / "phase_0" / "outputs" / "patient_split.csv"

VALID_SPLITS = ("train", "calibration", "test")


def _install_numpy_pickle_compat() -> None:
    """Allow NumPy 1.x to read pickles created by NumPy 2.x."""
    import numpy.core as np_core
    import numpy.core.multiarray as np_multiarray
    import numpy.core.numeric as np_numeric

    sys.modules.setdefault("numpy._core", np_core)
    sys.modules.setdefault("numpy._core.multiarray", np_multiarray)
    sys.modules.setdefault("numpy._core.numeric", np_numeric)


def _build_split_from_source() -> pd.DataFrame:
    if not SOURCE_DATA.exists():
        raise FileNotFoundError(
            f"Cannot build canonical patient split: source data missing at {SOURCE_DATA}"
        )
    _install_numpy_pickle_compat()
    df = pd.read_pickle(SOURCE_DATA)
    if "PatientId" not in df.columns:
        raise ValueError(f"PatientId column not found in {SOURCE_DATA}")

    patients = np.asarray(
        sorted(df["PatientId"].astype(str).drop_duplicates().tolist()),
        dtype=object,
    )
    rng = np.random.default_rng(RANDOM_SEED)

    test_size = int(round(len(patients) * TEST_PATIENT_FRACTION))
    test_patients = set(rng.choice(patients, size=test_size, replace=False))

    remaining = np.asarray(
        [p for p in patients if p not in test_patients], dtype=object
    )
    calibration_size = int(round(len(patients) * CALIBRATION_PATIENT_FRACTION))
    calibration_patients = set(
        rng.choice(remaining, size=calibration_size, replace=False)
    )

    splits = []
    for patient in patients:
        if patient in test_patients:
            splits.append("test")
        elif patient in calibration_patients:
            splits.append("calibration")
        else:
            splits.append("train")

    return pd.DataFrame({"PatientId": patients, "split": splits})


def load_canonical_split(*, rebuild: bool = False) -> pd.DataFrame:
    """Return the canonical (PatientId, split) frame, building the cache if needed."""
    if SPLIT_CACHE.exists() and not rebuild:
        cached = pd.read_csv(SPLIT_CACHE, dtype={"PatientId": str})
        if set(cached.columns) >= {"PatientId", "split"} and set(cached["split"]).issubset(VALID_SPLITS):
            return cached

    split_df = _build_split_from_source()
    SPLIT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(SPLIT_CACHE, index=False)
    return split_df


def _join_split(df: pd.DataFrame, split_df: pd.DataFrame) -> pd.Series:
    if "PatientId" not in df.columns:
        raise ValueError("Input frame must have a PatientId column")
    patient_ids = df["PatientId"].astype(str)
    mapping = split_df.set_index("PatientId")["split"]
    assignment = patient_ids.map(mapping)
    missing = assignment.isna()
    if missing.any():
        unknown = patient_ids[missing].drop_duplicates().head(5).tolist()
        raise ValueError(
            "Found PatientIds not in canonical split "
            f"(showing up to 5): {unknown}. Rebuild the canonical split if the "
            "underlying patient cohort has changed."
        )
    return assignment


def apply_canonical_split(
    df: pd.DataFrame,
    *,
    return_calibration: bool = False,
) -> tuple[pd.DataFrame, ...]:
    """Split ``df`` into (train, test) or (train, calibration, test) by PatientId.

    The returned frames are .copy() of the original rows preserving original order.
    When ``return_calibration`` is False, calibration patients are folded into train.
    """
    split_df = load_canonical_split()
    assignment = _join_split(df, split_df)

    test_mask = assignment.eq("test").to_numpy()
    cal_mask = assignment.eq("calibration").to_numpy()
    train_mask = ~(test_mask | cal_mask)

    test_df = df.loc[test_mask].copy()
    if return_calibration:
        cal_df = df.loc[cal_mask].copy()
        train_df = df.loc[train_mask].copy()
        return train_df, cal_df, test_df

    train_df = df.loc[~test_mask].copy()
    return train_df, test_df


def split_summary() -> pd.DataFrame:
    """Return per-split row counts of the canonical split itself (for reporting)."""
    split_df = load_canonical_split()
    counts = split_df["split"].value_counts().reindex(VALID_SPLITS, fill_value=0)
    total = int(counts.sum())
    return pd.DataFrame(
        {
            "split": counts.index.tolist(),
            "patients": counts.to_numpy(dtype=int),
            "fraction": (counts.to_numpy(dtype=float) / max(total, 1)).round(4),
        }
    )


def patients_in(*splits: str) -> set[str]:
    """Return the set of PatientIds belonging to one or more canonical splits."""
    invalid = [s for s in splits if s not in VALID_SPLITS]
    if invalid:
        raise ValueError(f"Unknown splits: {invalid}; expected subset of {VALID_SPLITS}")
    split_df = load_canonical_split()
    return set(split_df.loc[split_df["split"].isin(splits), "PatientId"].astype(str))


def assert_known_patients(patient_ids: Iterable[str]) -> None:
    known = set(load_canonical_split()["PatientId"].astype(str))
    unknown = [p for p in (str(p) for p in patient_ids) if p not in known]
    if unknown:
        raise ValueError(
            f"PatientIds missing from canonical split: {unknown[:5]} (and more)"
            if len(unknown) > 5
            else f"PatientIds missing from canonical split: {unknown}"
        )
