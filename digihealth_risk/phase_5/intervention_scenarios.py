"""Generate Phase 6 v2 intervention scenario risk-score tables.

v2 applies Phase 0.2 feature engineering (FBS hinges, FBS_x_Age, MAX_FBS_x_Age)
and re-computes derived features after each preset is applied.

Examples:
    python digihealth_risk/phase_5/intervention_scenarios.py \
      --patient-id "76562/29" --horizons 1 3 5

    python digihealth_risk/phase_5/intervention_scenarios.py \
      --max-patients 100 --horizons 3 --preset combined_lifestyle
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from digihealth_risk.phase_2.train_tree_models import RANDOM_SEED, engineer_features, load_table  # noqa: E402

OUT_DIR = ROOT / "digihealth_risk" / "phase_5" / "outputs"
_PHASE0_OUT = ROOT / "digihealth_risk" / "phase_0" / "outputs"
_MODEL_DIR = ROOT / "digihealth_risk" / "phase_5" / "outputs" / "models_v2"
HISTORY_YEARS = 5


def phase0_path(horizon: int) -> Path:
    p = _PHASE0_OUT / f"phase_0_modeling_table_horizon_{horizon}_history_{HISTORY_YEARS}.pkl"
    if p.exists():
        return p
    d = _PHASE0_OUT / "phase_0_modeling_table.pkl"
    if horizon == 1 and d.exists():
        return d
    raise FileNotFoundError(f"No Phase 0 table for horizon={horizon}, history={HISTORY_YEARS}.")


def model_path(horizon: int) -> Path:
    return _MODEL_DIR / f"phase6_v2_monotonic_xgboost_n{horizon}_m{HISTORY_YEARS}.joblib"
DEFAULT_HORIZONS = [1, 2, 3, 4, 5]
TOLERANCE = 1e-10


def predict_probability(artifact: dict[str, Any], row: pd.DataFrame) -> float:
    x = artifact["preprocessor"].transform(row[artifact["feature_columns"]].copy())
    return float(artifact["model"].predict_proba(x)[:, 1][0])


def risk_score(probability: float) -> float:
    return float(np.clip(probability * 100.0, 0.0, 100.0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Phase 6 v2 intervention scenario scores.")
    parser.add_argument("--patient-id", action="append", help="PatientId to score. Can be repeated.")
    parser.add_argument("--source-year", type=int, help="Source year T. Defaults to latest eligible row per patient.")
    parser.add_argument("--horizons", nargs="+", type=int, default=DEFAULT_HORIZONS, choices=DEFAULT_HORIZONS)
    parser.add_argument(
        "--preset",
        action="append",
        choices=list(PRESET_REGISTRY),
        help="Intervention preset to run. Can be repeated. Defaults to all presets.",
    )
    parser.add_argument(
        "--max-patients",
        type=int,
        help="If --patient-id is omitted, sample this many patients per horizon from eligible modeling rows.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=OUT_DIR / "phase_6_v2_intervention_scenarios.csv",
        help="CSV output path.",
    )
    return parser.parse_args()


def numeric_clip(row: pd.DataFrame, feature: str, value: float, artifact: dict[str, Any]) -> float:
    ranges = artifact.get("train_feature_ranges", {}).get(feature, {})
    minimum = ranges.get("min")
    maximum = ranges.get("max")
    if minimum is not None and maximum is not None:
        return float(np.clip(value, minimum, maximum))
    return float(value)


def set_feature(row: pd.DataFrame, feature: str, value: float, artifact: dict[str, Any]) -> None:
    if feature in row.columns:
        row.loc[:, feature] = numeric_clip(row, feature, value, artifact)


def set_min(row: pd.DataFrame, feature: str, artifact: dict[str, Any]) -> None:
    ranges = artifact.get("train_feature_ranges", {}).get(feature, {})
    if feature in row.columns and ranges.get("min") is not None:
        row.loc[:, feature] = float(ranges["min"])


def set_p75(row: pd.DataFrame, feature: str, artifact: dict[str, Any], reference_df: pd.DataFrame) -> None:
    if feature in row.columns:
        value = float(reference_df[feature].quantile(0.75))
        row.loc[:, feature] = np.maximum(row[feature].astype(float), numeric_clip(row, feature, value, artifact))


def reduce_sugary_50(row: pd.DataFrame, artifact: dict[str, Any], _: pd.DataFrame) -> pd.DataFrame:
    adjusted = row.copy()
    if "total_sugary_week" in adjusted.columns:
        set_feature(adjusted, "total_sugary_week", float(adjusted["total_sugary_week"].iloc[0]) * 0.5, artifact)
    return adjusted


def reduce_sugary_zero(row: pd.DataFrame, artifact: dict[str, Any], _: pd.DataFrame) -> pd.DataFrame:
    adjusted = row.copy()
    set_min(adjusted, "total_sugary_week", artifact)
    return adjusted


def exercise_p75(row: pd.DataFrame, artifact: dict[str, Any], reference_df: pd.DataFrame) -> pd.DataFrame:
    adjusted = row.copy()
    set_p75(adjusted, "total_exercise_week", artifact, reference_df)
    return adjusted


def activity_p75(row: pd.DataFrame, artifact: dict[str, Any], reference_df: pd.DataFrame) -> pd.DataFrame:
    adjusted = row.copy()
    set_p75(adjusted, "total_phy_activity_week", artifact, reference_df)
    return adjusted


def veg_fruit_p75(row: pd.DataFrame, artifact: dict[str, Any], reference_df: pd.DataFrame) -> pd.DataFrame:
    adjusted = row.copy()
    set_p75(adjusted, "total_veg_fruit_week", artifact, reference_df)
    return adjusted


def bmi_minus_one(row: pd.DataFrame, artifact: dict[str, Any], _: pd.DataFrame) -> pd.DataFrame:
    adjusted = row.copy()
    if "BMI" in adjusted.columns and pd.notna(adjusted["BMI"].iloc[0]):
        set_feature(adjusted, "BMI", float(adjusted["BMI"].iloc[0]) - 1.0, artifact)
    return adjusted


def combined_lifestyle(row: pd.DataFrame, artifact: dict[str, Any], reference_df: pd.DataFrame) -> pd.DataFrame:
    adjusted = reduce_sugary_50(row, artifact, reference_df)
    adjusted = exercise_p75(adjusted, artifact, reference_df)
    adjusted = activity_p75(adjusted, artifact, reference_df)
    adjusted = veg_fruit_p75(adjusted, artifact, reference_df)
    return adjusted


PRESET_REGISTRY: dict[str, Callable[[pd.DataFrame, dict[str, Any], pd.DataFrame], pd.DataFrame]] = {
    "reduce_sugary_50": reduce_sugary_50,
    "reduce_sugary_zero": reduce_sugary_zero,
    "exercise_p75": exercise_p75,
    "activity_p75": activity_p75,
    "veg_fruit_p75": veg_fruit_p75,
    "bmi_minus_one": bmi_minus_one,
    "combined_lifestyle": combined_lifestyle,
}

FAVORABLE_PRESETS = set(PRESET_REGISTRY)


def latest_rows_for_patients(df: pd.DataFrame, patient_ids: list[str] | None, source_year: int | None) -> pd.DataFrame:
    eligible = df.copy()
    if patient_ids:
        eligible = eligible[eligible["PatientId"].astype(str).isin([str(pid) for pid in patient_ids])].copy()
    if source_year is not None:
        eligible = eligible[eligible["Year"].eq(source_year)].copy()
    if eligible.empty:
        raise ValueError("No eligible modeling rows found for requested patient/source-year selection.")
    return (
        eligible.sort_values(["PatientId", "Year"])
        .groupby("PatientId", as_index=False)
        .tail(1)
        .sort_values(["PatientId", "Year"])
        .reset_index(drop=True)
    )


def sampled_latest_rows(df: pd.DataFrame, max_patients: int | None, source_year: int | None) -> pd.DataFrame:
    rows = latest_rows_for_patients(df, None, source_year)
    if max_patients is not None and len(rows) > max_patients:
        rows = rows.sample(n=max_patients, random_state=RANDOM_SEED).sort_values(["PatientId", "Year"])
    return rows.reset_index(drop=True)


def changed_feature_summary(baseline: pd.DataFrame, adjusted: pd.DataFrame) -> str:
    changes = []
    for column in adjusted.columns:
        if column not in baseline.columns:
            continue
        before = baseline[column].iloc[0]
        after = adjusted[column].iloc[0]
        if pd.isna(before) and pd.isna(after):
            continue
        if str(before) != str(after):
            changes.append(f"{column}:{before}->{after}")
    return "; ".join(changes)


def clipped_feature_summary(baseline: pd.DataFrame, adjusted: pd.DataFrame, artifact: dict[str, Any]) -> str:
    clipped = []
    ranges = artifact.get("train_feature_ranges", {})
    for feature, bounds in ranges.items():
        if feature not in adjusted.columns or bounds.get("min") is None or bounds.get("max") is None:
            continue
        before = baseline[feature].iloc[0] if feature in baseline.columns else np.nan
        value = adjusted[feature].iloc[0]
        if pd.isna(value):
            continue
        if pd.isna(before) or str(before) == str(value):
            continue
        if float(value) <= float(bounds["min"]) + TOLERANCE:
            clipped.append(f"{feature}=min({bounds['min']})")
        elif float(value) >= float(bounds["max"]) - TOLERANCE:
            clipped.append(f"{feature}=max({bounds['max']})")
    return "; ".join(clipped)


def normalize_model_row(row: pd.DataFrame) -> pd.DataFrame:
    """Keep row-wise Series conversion from turning numeric missing values into pd.NA objects."""
    return row.replace({pd.NA: np.nan})


def score_rows(
    horizon: int,
    rows: pd.DataFrame,
    presets: list[str],
) -> pd.DataFrame:
    artifact = joblib.load(model_path(horizon))
    reference_df = load_table(phase0_path(horizon))
    reference_df = engineer_features(reference_df)  # v2: ensure reference has derived features
    records = []

    for _, base_row in rows.iterrows():
        baseline = normalize_model_row(base_row.to_frame().T.copy())
        baseline_probability = predict_probability(artifact, baseline)
        baseline_score = risk_score(baseline_probability)

        records.append(
            {
                "PatientId": baseline["PatientId"].iloc[0],
                "source_year": int(baseline["Year"].iloc[0]),
                "target_year": int(baseline["target_year"].iloc[0]),
                "horizon_years": horizon,
                "scenario": "baseline",
                "baseline_score": baseline_score,
                "scenario_score": baseline_score,
                "delta_score": 0.0,
                "changed_features": "",
                "clipped_features": "",
                "no_effect": False,
                "unexpected_increase": False,
                "warning": "",
            }
        )

        for preset in presets:
            adjusted = PRESET_REGISTRY[preset](baseline, artifact, reference_df)
            # v2: recompute FBS-derived features after preset modifies base features.
            # Year_centered is already in the row so engineer_features preserves it.
            adjusted = engineer_features(adjusted)
            scenario_probability = predict_probability(artifact, adjusted)
            scenario_score = risk_score(scenario_probability)
            delta_score = scenario_score - baseline_score
            changed_features = changed_feature_summary(baseline, adjusted)
            no_effect = abs(delta_score) <= TOLERANCE
            unexpected_increase = preset in FAVORABLE_PRESETS and delta_score > TOLERANCE
            warnings = []
            if not changed_features:
                warnings.append("no_feature_change")
            if no_effect:
                warnings.append("no_score_change")
            if unexpected_increase:
                warnings.append("unexpected_risk_increase")
            records.append(
                {
                    "PatientId": baseline["PatientId"].iloc[0],
                    "source_year": int(baseline["Year"].iloc[0]),
                    "target_year": int(baseline["target_year"].iloc[0]),
                    "horizon_years": horizon,
                    "scenario": preset,
                    "baseline_score": baseline_score,
                    "scenario_score": scenario_score,
                    "delta_score": delta_score,
                    "changed_features": changed_features,
                    "clipped_features": clipped_feature_summary(baseline, adjusted, artifact),
                    "no_effect": no_effect,
                    "unexpected_increase": unexpected_increase,
                    "warning": "; ".join(warnings),
                }
            )
    return pd.DataFrame(records)


def write_summary(result: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    summary = (
        result[result["scenario"].ne("baseline")]
        .groupby(["horizon_years", "scenario"], as_index=False)
        .agg(
            rows=("delta_score", "size"),
            mean_delta_score=("delta_score", "mean"),
            min_delta_score=("delta_score", "min"),
            max_delta_score=("delta_score", "max"),
            no_effect_rows=("no_effect", "sum"),
            unexpected_increase_rows=("unexpected_increase", "sum"),
        )
    )
    summary_path = output_path.with_name(output_path.stem + "_summary.csv")
    summary.to_csv(summary_path, index=False)
    return summary


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    presets = args.preset or list(PRESET_REGISTRY)

    parts = []
    for horizon in sorted(set(args.horizons)):
        table = load_table(phase0_path(horizon))
        table = engineer_features(table)  # v2: apply before selecting rows
        if args.patient_id:
            rows = latest_rows_for_patients(table, args.patient_id, args.source_year)
        else:
            rows = sampled_latest_rows(table, args.max_patients, args.source_year)
        parts.append(score_rows(horizon, rows, presets))

    result = pd.concat(parts, ignore_index=True)
    result.to_csv(args.output_path, index=False)
    summary = write_summary(result, args.output_path)

    unexpected = int(summary["unexpected_increase_rows"].sum()) if not summary.empty else 0
    if unexpected:
        raise RuntimeError(
            f"Found {unexpected} favorable intervention rows with increased risk. "
            "Review monotonic constraints and preset feature directions."
        )

    print(f"Wrote {len(result):,} scenario rows to {args.output_path}")
    if args.patient_id:
        display = result.copy()
        for column in ["baseline_score", "scenario_score", "delta_score"]:
            display[column] = display[column].map(lambda value: f"{value:.2f}")
        print(display.to_string(index=False))


if __name__ == "__main__":
    main()
