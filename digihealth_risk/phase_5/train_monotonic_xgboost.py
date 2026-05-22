"""Phase 6 v2 intervention-ready monotonic XGBoost risk models.

Applies Phase 0.2 EDA feature improvements over v1:
  - pulse_pressure removed (VIF=225)
  - FBS_hinge_100, FBS_hinge_125 added (clinical pre-DM/DM thresholds)
  - Year_centered_sq added (U-shaped temporal trend)
  - FBS_x_Age, MAX_FBS_x_Age added (Phase 0.2 top cross-lag interactions)

Monotonic constraints are extended to cover the new FBS-derived features
(FBS_hinge_100, FBS_hinge_125, FBS_x_Age, MAX_FBS_x_Age all have sign +1).

Run from the repository root:
    python digihealth_risk/phase_5/train_monotonic_xgboost.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from digihealth_risk.phase_2.train_tree_models import (  # noqa: E402
    engineer_features,
    get_feature_columns,
    RANDOM_SEED,
    classification_metrics,
    load_table,
    make_preprocessor,
    split_by_patient,
)

PHASE0_OUT = ROOT / "digihealth_risk" / "phase_0" / "outputs"
OUT_DIR = ROOT / "digihealth_risk" / "phase_5" / "outputs"
MODEL_DIR = OUT_DIR / "models_v2"
HORIZONS = [1, 2, 3, 4, 5]
HISTORY_YEARS = 5
SANITY_SAMPLE_SIZE = 5000
TOLERANCE = 1e-10


@dataclass(frozen=True)
class MonotoneRule:
    sign: int
    reason: str


BASE_MONOTONE_RULES: dict[str, MonotoneRule] = {
    "total_sugary_week": MonotoneRule(1, "Higher sugary drink intake should not lower risk."),
    "total_veg_fruit_week": MonotoneRule(-1, "Higher vegetable/fruit intake should not raise risk."),
    "total_exercise_week": MonotoneRule(-1, "Higher exercise should not raise risk."),
    "total_phy_activity_week": MonotoneRule(-1, "Higher physical activity should not raise risk."),
    "FBS": MonotoneRule(1, "Higher current fasting blood sugar should not lower risk."),
    "MAX_FBS_up_to_year": MonotoneRule(1, "Higher cumulative maximum fasting blood sugar should not lower risk."),
    "BMI": MonotoneRule(1, "Higher BMI should not lower risk."),
    "Waist": MonotoneRule(1, "Higher waist circumference should not lower risk."),
    # v2: FBS-derived features — same increasing-risk direction as FBS
    "FBS_hinge_100": MonotoneRule(1, "FBS excess above 100 mg/dL (pre-DM threshold) should not lower risk."),
    "FBS_hinge_125": MonotoneRule(1, "FBS excess above 125 mg/dL (DM threshold) should not lower risk."),
    "FBS_x_Age": MonotoneRule(1, "FBS × Age interaction — higher product should not lower risk."),
    "MAX_FBS_x_Age": MonotoneRule(1, "MAX_FBS_up_to_year × Age interaction — higher product should not lower risk."),
}

HISTORY_MONOTONE_BASES = {
    "FBS": 1,
    "BMI": 1,
    "Waist": 1,
}

HISTORY_MONOTONE_SUFFIXES = (
    "_latest",
    "_mean",
    "_min",
    "_max",
    "_range",
    "_slope",
)

INTERVENTION_POLICIES = {
    "reduce_sugary_to_zero": ("total_sugary_week", "set_min", "decrease_or_equal"),
    "increase_exercise_to_p75": ("total_exercise_week", "set_p75", "decrease_or_equal"),
    "increase_activity_to_p75": ("total_phy_activity_week", "set_p75", "decrease_or_equal"),
    "increase_veg_fruit_to_p75": ("total_veg_fruit_week", "set_p75", "decrease_or_equal"),
    "reduce_bmi_by_one": ("BMI", "subtract_one", "decrease_or_equal"),
    "increase_fbs_by_ten": ("FBS", "add_ten", "increase_or_equal"),
}


def phase0_path(horizon: int) -> Path:
    return PHASE0_OUT / f"phase_0_modeling_table_horizon_{horizon}_history_{HISTORY_YEARS}.pkl"


def rule_for_feature(feature_name: str) -> MonotoneRule | None:
    if feature_name in BASE_MONOTONE_RULES:
        return BASE_MONOTONE_RULES[feature_name]

    for base, sign in HISTORY_MONOTONE_BASES.items():
        prefix = f"{base}_hist_{HISTORY_YEARS}y"
        if feature_name.startswith(prefix) and feature_name.endswith(HISTORY_MONOTONE_SUFFIXES):
            return MonotoneRule(sign, f"History feature follows {base} monotonic direction.")

    return None


def monotone_constraints(feature_names: list[str]) -> tuple[int, ...]:
    constraints = []
    for feature in feature_names:
        rule = rule_for_feature(feature)
        constraints.append(0 if rule is None else rule.sign)
    return tuple(constraints)


def constraint_table(feature_names: list[str], constraints: tuple[int, ...], horizon: int) -> pd.DataFrame:
    rows = []
    for feature, sign in zip(feature_names, constraints, strict=True):
        rule = rule_for_feature(feature)
        rows.append(
            {
                "horizon_years": horizon,
                "feature": feature,
                "constraint": sign,
                "direction": "increasing_risk" if sign == 1 else "decreasing_risk" if sign == -1 else "none",
                "reason": "" if rule is None else rule.reason,
            }
        )
    return pd.DataFrame(rows)


def build_xgboost(constraints: tuple[int, ...]) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=400,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=2.0,
        objective="binary:logistic",
        eval_metric="logloss",
        monotone_constraints=constraints,
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )


def fit_monotonic_model(train_df: pd.DataFrame) -> dict[str, Any]:
    numeric_features, categorical_features = get_feature_columns(train_df)
    feature_columns = numeric_features + categorical_features
    preprocessor = make_preprocessor(numeric_features, categorical_features)
    x_train = preprocessor.fit_transform(train_df[feature_columns].copy())
    transformed_names = [str(name) for name in preprocessor.get_feature_names_out()]
    constraints = monotone_constraints(transformed_names)
    model = build_xgboost(constraints)
    y_train = train_df["Target_AtRisk_Status"].astype(int).to_numpy()
    model.fit(x_train, y_train)
    return {
        "preprocessor": preprocessor,
        "model": model,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "feature_columns": feature_columns,
        "transformed_feature_names": transformed_names,
        "monotone_constraints": constraints,
    }


def predict_probability(artifact: dict[str, Any], df: pd.DataFrame) -> np.ndarray:
    x = artifact["preprocessor"].transform(df[artifact["feature_columns"]].copy())
    return artifact["model"].predict_proba(x)[:, 1]


def apply_policy(df: pd.DataFrame, policy: tuple[str, str, str], train_df: pd.DataFrame) -> pd.DataFrame:
    feature, action, _ = policy
    adjusted = df.copy()
    if feature not in adjusted.columns:
        return adjusted

    if action == "set_min":
        value = float(train_df[feature].min(skipna=True))
        adjusted[feature] = np.minimum(adjusted[feature], value)
    elif action == "set_p75":
        value = float(train_df[feature].quantile(0.75))
        adjusted[feature] = np.maximum(adjusted[feature], value)
    elif action == "subtract_one":
        minimum = float(train_df[feature].min(skipna=True))
        adjusted[feature] = np.maximum(adjusted[feature] - 1.0, minimum)
    elif action == "add_ten":
        maximum = float(train_df[feature].max(skipna=True))
        adjusted[feature] = np.minimum(adjusted[feature] + 10.0, maximum)
    else:
        raise ValueError(f"Unsupported intervention action: {action}")

    # v2: recompute FBS-derived features when FBS changes
    adjusted = engineer_features(adjusted)
    return adjusted


def monotonic_sanity_checks(artifact: dict[str, Any], train_df: pd.DataFrame, test_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    sample = test_df.sample(n=min(SANITY_SAMPLE_SIZE, len(test_df)), random_state=RANDOM_SEED)
    baseline = predict_probability(artifact, sample)
    rows = []

    for policy_name, policy in INTERVENTION_POLICIES.items():
        _, _, expected = policy
        adjusted = apply_policy(sample, policy, train_df)
        adjusted_probability = predict_probability(artifact, adjusted)
        delta = adjusted_probability - baseline

        if expected == "decrease_or_equal":
            violations = int((delta > TOLERANCE).sum())
        elif expected == "increase_or_equal":
            violations = int((delta < -TOLERANCE).sum())
        else:
            raise ValueError(f"Unsupported expected direction: {expected}")

        rows.append(
            {
                "horizon_years": horizon,
                "policy": policy_name,
                "expected_direction": expected,
                "rows_checked": float(len(sample)),
                "violations": float(violations),
                "violation_rate": float(violations / max(len(sample), 1)),
                "mean_delta_probability": float(delta.mean()),
                "min_delta_probability": float(delta.min()),
                "max_delta_probability": float(delta.max()),
            }
        )
    return pd.DataFrame(rows)


def risk_score(probability: np.ndarray) -> np.ndarray:
    return np.clip(probability * 100.0, 0.0, 100.0)


def run_horizon(horizon: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = load_table(phase0_path(horizon))
    df = engineer_features(df)  # v2: add FBS hinges, Year_centered_sq, FBS_x_Age, MAX_FBS_x_Age
    train_df, test_df = split_by_patient(df)
    artifact = fit_monotonic_model(train_df)
    train_probability = predict_probability(artifact, train_df)
    test_probability = predict_probability(artifact, test_df)
    y_train = train_df["Target_AtRisk_Status"].astype(int).to_numpy()
    y_test = test_df["Target_AtRisk_Status"].astype(int).to_numpy()
    threshold = float(y_train.mean())

    metrics = pd.DataFrame(
        [
            {
                "horizon_years": horizon,
                "history_years": HISTORY_YEARS,
                "model_key": f"phase6_v2_monotonic_xgboost_n{horizon}_m{HISTORY_YEARS}",
                "model_name": "monotonic_xgboost_v2",
                "split": "train",
                **classification_metrics(y_train, train_probability, threshold),
            },
            {
                "horizon_years": horizon,
                "history_years": HISTORY_YEARS,
                "model_key": f"phase6_v2_monotonic_xgboost_n{horizon}_m{HISTORY_YEARS}",
                "model_name": "monotonic_xgboost_v2",
                "split": "test",
                **classification_metrics(y_test, test_probability, threshold),
            },
        ]
    )

    predictions = test_df[["PatientId", "Year", "target_year", "Target_AtRisk_Status"]].copy()
    predictions["horizon_years"] = horizon
    predictions["history_years"] = HISTORY_YEARS
    predictions["model_key"] = f"phase6_v2_monotonic_xgboost_n{horizon}_m{HISTORY_YEARS}"
    predictions["predicted_probability"] = test_probability
    predictions["risk_score_0_100"] = risk_score(test_probability)

    constraints = constraint_table(
        artifact["transformed_feature_names"],
        artifact["monotone_constraints"],
        horizon,
    )
    sanity = monotonic_sanity_checks(artifact, train_df, test_df, horizon)

    artifact.update(
        {
            "horizon_years": horizon,
            "history_years": HISTORY_YEARS,
            "model_key": f"phase6_v2_monotonic_xgboost_n{horizon}_m{HISTORY_YEARS}",
            "threshold": threshold,
            "train_positive_rate": threshold,
            "train_feature_ranges": {
                column: {
                    "min": float(train_df[column].min(skipna=True)) if pd.api.types.is_numeric_dtype(train_df[column]) else None,
                    "max": float(train_df[column].max(skipna=True)) if pd.api.types.is_numeric_dtype(train_df[column]) else None,
                }
                for column in artifact["feature_columns"]
            },
        }
    )
    joblib.dump(artifact, MODEL_DIR / f"phase6_v2_monotonic_xgboost_n{horizon}_m{HISTORY_YEARS}.joblib")

    return metrics, predictions, constraints, sanity


def markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if max_rows is not None:
        df = df.head(max_rows)
    display = df.copy()
    for column in display.select_dtypes(include=[np.number]).columns:
        display[column] = display[column].map(lambda x: "" if pd.isna(x) else f"{x:.4g}")
    columns = display.columns.tolist()
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def write_report(metrics: pd.DataFrame, sanity: pd.DataFrame, constraints: pd.DataFrame) -> str:
    test = metrics[metrics["split"].eq("test")].copy()
    metric_cols = [
        "horizon_years",
        "rows",
        "positives",
        "positive_rate",
        "roc_auc",
        "pr_auc",
        "brier",
        "precision",
        "recall",
        "specificity",
        "f1",
    ]
    constrained = constraints[constraints["constraint"].ne(0)].copy()
    lines = [
        "# Phase 6 v2 Intervention-Ready Risk Score Report",
        "",
        "## Scope",
        "This phase trains monotonic XGBoost models for `N=1..5,M=5` with v2 feature engineering. "
        "These models are designed for intervention-safe what-if scoring, not as replacements for the unconstrained best-prediction leaderboard.",
        "",
        "## v2 Feature Changes",
        "| Change | Feature | Reason |",
        "| --- | --- | --- |",
        "| Removed | `pulse_pressure` | VIF=225; equals BL_pres1 − BL_pres2 by construction |",
        "| Added | `FBS_hinge_100` | Hockey-stick at pre-DM threshold (100 mg/dL); constrained sign=+1 |",
        "| Added | `FBS_hinge_125` | Hockey-stick at DM threshold (125 mg/dL); constrained sign=+1 |",
        "| Added | `Year_centered_sq` | U-shaped temporal risk trend (Ljung-Box p=0.03); unconstrained |",
        "| Added | `FBS_x_Age` | Phase 0.2 top cross-lag interaction; constrained sign=+1 |",
        "| Added | `MAX_FBS_x_Age` | MAX_FBS_up_to_year × Age (cross-lag r=0.582); constrained sign=+1 |",
        "",
        "## Test Metrics",
        markdown_table(test[metric_cols]),
        "",
        "## Monotonic Sanity Checks",
        markdown_table(sanity),
        "",
        "## Constrained Features",
        markdown_table(constrained[["feature", "constraint", "direction", "reason"]].drop_duplicates(), max_rows=80),
        "",
        "## Notes",
        "- Risk score is `predicted_probability * 100` for a fixed horizon.",
        "- Monotonic constraints enforce directionality but do not prove causal effect size.",
        "- After any intervention on FBS, derived features (FBS_hinge_100, FBS_hinge_125, FBS_x_Age) are recomputed before scoring.",
        "- Use unconstrained Phase 4/4.2 models for best pure prediction; use Phase 6 v2 models for user-facing what-if intervention simulations.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    metric_parts = []
    prediction_parts = []
    constraint_parts = []
    sanity_parts = []
    for horizon in HORIZONS:
        print(f"Training monotonic XGBoost v2 N={horizon}, M={HISTORY_YEARS}")
        metrics, predictions, constraints, sanity = run_horizon(horizon)
        metric_parts.append(metrics)
        prediction_parts.append(predictions)
        constraint_parts.append(constraints)
        sanity_parts.append(sanity)

    metrics_df = pd.concat(metric_parts, ignore_index=True)
    predictions_df = pd.concat(prediction_parts, ignore_index=True)
    constraints_df = pd.concat(constraint_parts, ignore_index=True)
    sanity_df = pd.concat(sanity_parts, ignore_index=True)

    metrics_df.to_csv(OUT_DIR / "phase_6_v2_monotonic_xgboost_metrics.csv", index=False)
    predictions_df.to_csv(OUT_DIR / "phase_6_v2_monotonic_xgboost_test_predictions.csv", index=False)
    constraints_df.to_csv(OUT_DIR / "phase_6_v2_monotonic_constraints.csv", index=False)
    sanity_df.to_csv(OUT_DIR / "phase_6_v2_monotonic_sanity_checks.csv", index=False)

    # Write alias files that lightgbm/catboost/ebm/logistic sibling scripts depend on.
    ablation_metrics = metrics_df.copy()
    ablation_metrics.insert(0, "variant", "monotonic")
    ablation_metrics.to_csv(OUT_DIR / "phase_6_v2_ablation_metrics.csv", index=False)

    # Aggregate per-policy sanity checks into the safety-summary schema.
    agg = (
        sanity_df[sanity_df["expected_direction"].eq("decrease_or_equal")]
        .groupby("horizon_years", as_index=False)
        .agg(
            unexpected_increase_rate=("violation_rate", "mean"),
            mean_delta_score=("mean_delta_probability", lambda s: (s * 100).mean()),
            worst_positive_delta_score=("max_delta_probability", lambda s: (s * 100).max()),
        )
    )
    agg["directionally_correct_rate"] = 1.0 - agg["unexpected_increase_rate"]
    agg.insert(0, "variant", "monotonic")
    agg["history_years"] = HISTORY_YEARS
    agg.to_csv(OUT_DIR / "phase_6_v2_ablation_safety_summary.csv", index=False)

    report = write_report(metrics_df, sanity_df, constraints_df)
    (OUT_DIR / "phase_6_v2_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
