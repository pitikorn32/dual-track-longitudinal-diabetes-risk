"""Reproduce the data-level and GEE interpretability results in the paper.

Run from the submodule root::

    python digihealth_risk/phase_4/feature_effects_analysis.py

The script reads the canonical one-year, one-year-history modeling table and
the saved five-year GEE coefficients. It writes two ignored CSV files under
``digihealth_risk/phase_4/outputs/``:

* ``phase_4_feature_effects.csv`` — univariate Cohen's *d* values;
* ``phase_4_gee_n5_m5_interpretability.csv`` — coefficients, robust standard
  errors, p-values, and significance flags for the five-year GEE winner.

The effect-size calculation matches the Phase 0 EDA definition: the pooled
standard deviation is computed from non-missing positive and negative target
rows, and Cohen's *d* is positive when the positive class has the larger mean.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODELING_TABLE = ROOT / "digihealth_risk" / "phase_0" / "outputs" / "phase_0_modeling_table.pkl"
DEFAULT_GEE_DIR = ROOT / "digihealth_risk" / "phase_1" / "outputs"
DEFAULT_OUTPUT_DIR = ROOT / "digihealth_risk" / "phase_4" / "outputs"
TARGET = "Target_AtRisk_Status"

NUMERIC_FEATURES = [
    "FBS",
    "BMI",
    "Pulse",
    "BL_pres1",
    "BL_pres2",
    "Waist",
    "total_sugary_week",
    "total_veg_fruit_week",
    "total_exercise_week",
    "total_phy_activity_week",
    "sleep_hours",
    "Age",
    "MAX_FBS_up_to_year",
    "clinical_observed_count",
    "has_fbs_this_year",
    "years_since_last_fbs",
]


def cohen_d(positive: np.ndarray, negative: np.ndarray) -> float:
    """Return pooled-standard-deviation Cohen's d for two non-empty groups."""
    n_positive, n_negative = len(positive), len(negative)
    if n_positive < 2 or n_negative < 2:
        return float("nan")
    pooled_variance = (
        (n_positive - 1) * positive.std(ddof=1) ** 2
        + (n_negative - 1) * negative.std(ddof=1) ** 2
    ) / (n_positive + n_negative - 2)
    pooled_std = np.sqrt(pooled_variance)
    return float((positive.mean() - negative.mean()) / pooled_std) if pooled_std > 0 else float("nan")


def compute_effects(modeling_table: pd.DataFrame) -> pd.DataFrame:
    """Compute univariate effect sizes against the modeling-table target."""
    target = modeling_table[TARGET].astype(int)
    positive_mask = target == 1
    negative_mask = target == 0
    rows: list[dict[str, object]] = []

    for feature in NUMERIC_FEATURES:
        if feature not in modeling_table.columns:
            continue
        column = modeling_table[feature]
        positive = column[positive_mask].dropna().to_numpy(dtype=float)
        negative = column[negative_mask].dropna().to_numpy(dtype=float)
        rows.append(
            {
                "feature": feature,
                "n_positive": len(positive),
                "n_negative": len(negative),
                "positive_mean": float(positive.mean()) if len(positive) else np.nan,
                "negative_mean": float(negative.mean()) if len(negative) else np.nan,
                "cohen_d_vs_target": cohen_d(positive, negative),
            }
        )

    return (
        pd.DataFrame(rows)
        .assign(abs_cohen_d=lambda frame: frame["cohen_d_vs_target"].abs())
        .sort_values("abs_cohen_d", ascending=False)
        .drop(columns="abs_cohen_d")
    )


def load_gee_interpretability(gee_dir: Path) -> pd.DataFrame:
    """Load the saved five-year, five-year-history GEE coefficients."""
    path = gee_dir / "phase_1_v2_gee_horizon_5_history_5_coefficients.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing GEE coefficient file: {path}")
    coefficients = pd.read_csv(path)
    coefficients["significant_0_05"] = coefficients["p_value"] < 0.05
    return coefficients


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modeling-table", type=Path, default=DEFAULT_MODELING_TABLE)
    parser.add_argument("--gee-dir", type=Path, default=DEFAULT_GEE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    modeling_table = pd.read_pickle(args.modeling_table)
    if TARGET not in modeling_table.columns:
        raise KeyError(f"{TARGET!r} is missing from {args.modeling_table}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    effects = compute_effects(modeling_table)
    gee = load_gee_interpretability(args.gee_dir)
    effects_path = args.output_dir / "phase_4_feature_effects.csv"
    gee_path = args.output_dir / "phase_4_gee_n5_m5_interpretability.csv"
    effects.to_csv(effects_path, index=False)
    gee.to_csv(gee_path, index=False)

    print(f"Wrote {effects_path}")
    print(effects.head(5).to_string(index=False))
    print(f"Wrote {gee_path}")
    print("Significant GEE features (p < 0.05):")
    print(gee.loc[gee["significant_0_05"], ["feature", "coefficient", "p_value"]].to_string(index=False))


if __name__ == "__main__":
    main()
