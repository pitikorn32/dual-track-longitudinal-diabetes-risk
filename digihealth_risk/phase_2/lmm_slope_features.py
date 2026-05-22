"""Phase 2 v2 hybrid slope feature extraction — improved feature engineering.

Changes from v1 (phase_2_2_lmm_slope_features.py) driven by Phase 0.2 EDA:
  - Removed `pulse_pressure` from SLOPE_FEATURES (VIF=225; = BL_pres1 − BL_pres2)
  - Added `FBS_hinge_100` to SLOPE_FEATURES (trend of pre-DM excess FBS)
  - Added engineer_features() to compute FBS_hinge_100 before slope extraction

Net: 7 slope features × 8 slope columns = 56 slope columns (same count as v1).

Run from the repository root:
    python digihealth_risk/phase_2/lmm_slope_features.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PHASE0_OUT = ROOT / "digihealth_risk" / "phase_0" / "outputs"
OUT_DIR = ROOT / "digihealth_risk" / "phase_2" / "outputs"
LONG_PATH = PHASE0_OUT / "patient_year_long.pkl"
DEFAULT_INPUT_PATHS = [
    PHASE0_OUT / "phase_0_modeling_table.pkl",
    PHASE0_OUT / "phase_0_modeling_table_horizon_3_history_5.pkl",
]
# pulse_pressure removed (VIF=225); FBS_hinge_100 added (pre-DM threshold trend)
SLOPE_FEATURES = ["FBS", "BMI", "Waist", "Pulse", "BL_pres1", "BL_pres2", "FBS_hinge_100"]
SHRINKAGE_STRENGTH = 3.0


def install_numpy_pickle_compat() -> None:
    import sys
    import numpy.core as np_core
    import numpy.core.multiarray as np_multiarray
    import numpy.core.numeric as np_numeric

    sys.modules.setdefault("numpy._core", np_core)
    sys.modules.setdefault("numpy._core.multiarray", np_multiarray)
    sys.modules.setdefault("numpy._core.numeric", np_numeric)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add Phase 2 v2 slope features.")
    parser.add_argument(
        "--input-path",
        type=Path,
        action="append",
        help="Phase 0 modeling table. Can be passed multiple times. Defaults to reviewed Phase 0 tables.",
    )
    parser.add_argument(
        "--shrinkage-strength",
        type=float,
        default=SHRINKAGE_STRENGTH,
        help="Higher values shrink sparse patient slopes more strongly toward population slopes.",
    )
    return parser.parse_args()


def output_stem(path: Path) -> str:
    stem = path.stem
    if stem == "phase_0_modeling_table":
        return "phase_2_v2_modeling_table_with_slopes"
    return stem.replace("phase_0", "phase_2_v2") + "_with_slopes"


def load_pickle(path: Path) -> pd.DataFrame:
    install_numpy_pickle_compat()
    return pd.read_pickle(path).copy()


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "FBS_hinge_100" not in df.columns:
        df["FBS_hinge_100"] = (df["FBS"] - 100).clip(lower=0)
    return df


def patient_series_map(long_df: pd.DataFrame) -> dict[str, dict[str, pd.Series]]:
    result: dict[str, dict[str, pd.Series]] = {}
    for feature in SLOPE_FEATURES:
        feature_map: dict[str, pd.Series] = {}
        for patient_id, group in long_df[["PatientId", "Year", feature]].groupby("PatientId", sort=False):
            series = group.set_index("Year")[feature].sort_index().dropna()
            feature_map[patient_id] = series
        result[feature] = feature_map
    return result


def fit_line(years: np.ndarray, values: np.ndarray, source_year: int) -> tuple[float, float, float]:
    x = years.astype(float) - float(source_year)
    y = values.astype(float)
    if len(np.unique(x)) < 2:
        return 0.0, float(np.nanmean(y)), np.nan
    slope, intercept_at_source = np.polyfit(x, y, deg=1)
    fitted = slope * x + intercept_at_source
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = np.nan if ss_tot == 0 else 1.0 - (ss_res / ss_tot)
    return float(slope), float(intercept_at_source), float(r2)


def population_trends(long_df: pd.DataFrame) -> dict[tuple[str, int], tuple[float, float]]:
    trends: dict[tuple[str, int], tuple[float, float]] = {}
    for feature in SLOPE_FEATURES:
        observed = long_df[["Year", feature]].dropna()
        for source_year in sorted(long_df["Year"].unique()):
            cutoff = observed[observed["Year"] <= source_year]
            if len(cutoff) < 2:
                trends[(feature, int(source_year))] = (0.0, float(cutoff[feature].mean()) if len(cutoff) else np.nan)
                continue
            slope, value_at_source, _ = fit_line(
                cutoff["Year"].to_numpy(),
                cutoff[feature].to_numpy(),
                int(source_year),
            )
            trends[(feature, int(source_year))] = (slope, value_at_source)
    return trends


def slope_features_for_row(
    patient_id: str,
    source_year: int,
    feature_map: dict[str, dict[str, pd.Series]],
    pop_trends: dict[tuple[str, int], tuple[float, float]],
    shrinkage_strength: float,
) -> dict[str, float]:
    record: dict[str, float] = {}

    for feature in SLOPE_FEATURES:
        series = feature_map[feature].get(patient_id, pd.Series(dtype=float))
        history = series[series.index <= source_year]
        n_points = int(history.size)
        pop_slope, pop_value_at_source = pop_trends[(feature, source_year)]

        ols_slope = np.nan
        ols_value_at_source = np.nan
        ols_r2 = np.nan
        if n_points >= 2:
            ols_slope, ols_value_at_source, ols_r2 = fit_line(
                history.index.to_numpy(),
                history.to_numpy(),
                source_year,
            )
        elif n_points == 1:
            ols_value_at_source = float(history.iloc[0])

        if np.isfinite(ols_slope):
            weight = (n_points - 1) / ((n_points - 1) + shrinkage_strength)
            shrunk_slope = weight * ols_slope + (1.0 - weight) * pop_slope
        else:
            weight = 0.0
            shrunk_slope = pop_slope

        if np.isfinite(ols_value_at_source):
            shrunk_value = weight * ols_value_at_source + (1.0 - weight) * pop_value_at_source
        else:
            shrunk_value = pop_value_at_source

        prefix = f"{feature}_trend"
        record[f"{prefix}_ols_slope_to_T"] = ols_slope
        record[f"{prefix}_ols_value_at_T"] = ols_value_at_source
        record[f"{prefix}_ols_r2_to_T"] = ols_r2
        record[f"{prefix}_points_to_T"] = n_points
        record[f"{prefix}_shrinkage_weight"] = weight
        record[f"{prefix}_lmm_shrunk_slope_to_T"] = shrunk_slope
        record[f"{prefix}_lmm_random_slope_to_T"] = shrunk_slope - pop_slope
        record[f"{prefix}_lmm_shrunk_value_at_T"] = shrunk_value

    return record


def add_slope_features(
    model_df: pd.DataFrame,
    feature_map: dict[str, dict[str, pd.Series]],
    pop_trends: dict[tuple[str, int], tuple[float, float]],
    shrinkage_strength: float,
) -> pd.DataFrame:
    records = [
        slope_features_for_row(
            patient_id=str(patient_id),
            source_year=int(source_year),
            feature_map=feature_map,
            pop_trends=pop_trends,
            shrinkage_strength=shrinkage_strength,
        )
        for patient_id, source_year in zip(model_df["PatientId"], model_df["Year"], strict=True)
    ]
    slope_df = pd.DataFrame(records, index=model_df.index)
    return pd.concat([model_df, slope_df], axis=1)


def summarize_outputs(outputs: list[tuple[Path, pd.DataFrame]], shrinkage_strength: float) -> str:
    lines = [
        "# Phase 2 v2 Hybrid Slope Feature Report",
        "",
        "## Scope",
        "Improved slope feature extraction with `pulse_pressure` removed (VIF=225) and "
        "`FBS_hinge_100` added to capture the trend of pre-DM excess FBS above 100 mg/dL. "
        "Each row at source year `T` uses only clinical observations from years `<= T`.",
        "",
        "## Slope Families",
        "- `*_ols_*`: patient-specific ordinary least-squares trend features.",
        "- `*_lmm_shrunk_*`: empirical-Bayes/shrinkage trend features that pull sparse patient slopes toward the population trend for the same cutoff year.",
        "",
        f"Shrinkage strength: `{shrinkage_strength}`.",
        "",
        "## v2 Changes vs v1",
        "| Change | Feature | Reason |",
        "| --- | --- | --- |",
        "| Removed | `pulse_pressure` | VIF=225; equals BL_pres1 − BL_pres2 by construction |",
        "| Added | `FBS_hinge_100` | Trend of pre-DM excess FBS; captures rate of rise above 100 mg/dL |",
        "",
        "## Clinical Features",
    ]
    for feature in SLOPE_FEATURES:
        lines.append(f"- `{feature}`")

    lines.extend(["", "## Outputs"])
    for path, df in outputs:
        slope_cols = [column for column in df.columns if "_trend_" in column]
        lines.append(
            f"- `{path.relative_to(ROOT)}`: `{df.shape[0]:,}` rows x `{df.shape[1]:,}` columns, "
            f"`{len(slope_cols)}` slope columns."
        )

    lines.extend(
        [
            "",
            "## Notes",
            "- True full mixed-effects refits for every feature/cutoff are computationally expensive on this dataset.",
            "- The shrinkage features approximate random-slope behavior for sparse longitudinal histories and are intended for tree-model comparison.",
            "- `*_lmm_random_slope_to_T` is the deviation from the cutoff-specific population slope.",
            "- `FBS_hinge_100` slope captures the rate of change of FBS in the pre-DM range (FBS > 100 mg/dL); zero for patients with consistently normal FBS.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    input_paths = args.input_path or [path for path in DEFAULT_INPUT_PATHS if path.exists()]
    if not input_paths:
        raise FileNotFoundError("No Phase 0 modeling tables found.")

    long_df = load_pickle(LONG_PATH)
    long_df = engineer_features(long_df)  # compute FBS_hinge_100 before slope extraction
    feature_map = patient_series_map(long_df)
    pop_trends = population_trends(long_df)

    outputs: list[tuple[Path, pd.DataFrame]] = []
    for input_path in input_paths:
        model_df = load_pickle(input_path)
        model_df = engineer_features(model_df)  # ensure FBS_hinge_100 present for downstream use
        enhanced = add_slope_features(model_df, feature_map, pop_trends, args.shrinkage_strength)
        output_path = OUT_DIR / f"{output_stem(input_path)}.pkl"
        sample_path = OUT_DIR / f"{output_stem(input_path)}_sample.csv"
        enhanced.to_pickle(output_path)
        enhanced.head(1000).to_csv(sample_path, index=False)
        outputs.append((output_path, enhanced))
        print(f"Wrote {output_path} shape={enhanced.shape}")

    report = summarize_outputs(outputs, args.shrinkage_strength)
    (OUT_DIR / "phase_2_v2_slope_feature_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
