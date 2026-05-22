"""Shared Phase 5 v2 helpers for monotonic intervention ablations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy import special

from digihealth_risk.phase_2.train_tree_models import engineer_features


ROOT = Path(__file__).resolve().parents[2]
PHASE0_OUT = ROOT / "digihealth_risk" / "phase_0" / "outputs"
OUT_DIR = ROOT / "digihealth_risk" / "phase_5" / "outputs"
PHASE4_RANKING_PATH = ROOT / "digihealth_risk" / "phase_4" / "outputs" / "phase_4_2_v2_cross_family_ranking.csv"
HORIZONS = [1, 2, 3, 4, 5]
HISTORY_OPTIONS = [3, 5]
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
    "FBS_hinge_100": MonotoneRule(1, "FBS excess above 100 mg/dL should not lower risk."),
    "FBS_hinge_125": MonotoneRule(1, "FBS excess above 125 mg/dL should not lower risk."),
    "FBS_x_Age": MonotoneRule(1, "FBS x Age should not lower risk."),
    "MAX_FBS_x_Age": MonotoneRule(1, "MAX_FBS_up_to_year x Age should not lower risk."),
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


def phase0_path(horizon: int, history_years: int) -> Path:
    path = PHASE0_OUT / f"phase_0_modeling_table_horizon_{horizon}_history_{history_years}.pkl"
    if path.exists():
        return path
    default = PHASE0_OUT / "phase_0_modeling_table.pkl"
    if horizon == 1 and history_years == 1 and default.exists():
        return default
    raise FileNotFoundError(f"No Phase 0 table for horizon={horizon}, history={history_years}.")


def risk_score(probability: np.ndarray) -> np.ndarray:
    return np.clip(probability * 100.0, 0.0, 100.0)


def rule_for_feature(feature_name: str, history_years: int) -> MonotoneRule | None:
    if feature_name in BASE_MONOTONE_RULES:
        return BASE_MONOTONE_RULES[feature_name]

    for base, sign in HISTORY_MONOTONE_BASES.items():
        prefix = f"{base}_hist_{history_years}y"
        if feature_name.startswith(prefix) and feature_name.endswith(HISTORY_MONOTONE_SUFFIXES):
            return MonotoneRule(sign, f"History feature follows {base} monotonic direction.")
    return None


def monotone_constraints(feature_names: list[str], history_years: int) -> tuple[int, ...]:
    constraints = []
    for feature in feature_names:
        rule = rule_for_feature(feature, history_years)
        constraints.append(0 if rule is None else rule.sign)
    return tuple(constraints)


def constraint_table(
    feature_names: list[str],
    constraints: tuple[int, ...],
    *,
    horizon: int,
    history_years: int,
    variant: str,
) -> pd.DataFrame:
    rows = []
    for feature, sign in zip(feature_names, constraints, strict=True):
        rule = rule_for_feature(feature, history_years)
        rows.append(
            {
                "variant": variant,
                "horizon_years": horizon,
                "history_years": history_years,
                "feature": feature,
                "constraint": sign,
                "direction": "increasing_risk" if sign == 1 else "decreasing_risk" if sign == -1 else "none",
                "reason": "" if rule is None else rule.reason,
            }
        )
    return pd.DataFrame(rows)


def predict_probability(artifact: dict[str, Any], df: pd.DataFrame) -> np.ndarray:
    x = artifact["preprocessor"].transform(df[artifact["feature_columns"]].copy())
    if {"coefficients", "mean_", "scale_"}.issubset(artifact):
        x_raw = np.asarray(x, dtype=float)
        x_scaled = (x_raw - artifact["mean_"]) / artifact["scale_"]
        x_design = np.hstack([np.ones((x_scaled.shape[0], 1), dtype=float), x_scaled])
        return special.expit(x_design @ artifact["coefficients"])
    return artifact["model"].predict_proba(x)[:, 1]


def numeric_clip(feature: str, value: float, artifact: dict[str, Any]) -> float:
    ranges = artifact.get("train_feature_ranges", {}).get(feature, {})
    minimum = ranges.get("min")
    maximum = ranges.get("max")
    if minimum is not None and maximum is not None:
        return float(np.clip(value, minimum, maximum))
    return float(value)


def set_feature(df: pd.DataFrame, feature: str, value: float, artifact: dict[str, Any]) -> None:
    if feature in df.columns:
        df.loc[:, feature] = numeric_clip(feature, value, artifact)


def set_min(df: pd.DataFrame, feature: str, artifact: dict[str, Any]) -> None:
    ranges = artifact.get("train_feature_ranges", {}).get(feature, {})
    if feature in df.columns and ranges.get("min") is not None:
        df.loc[:, feature] = float(ranges["min"])


def set_p75(df: pd.DataFrame, feature: str, artifact: dict[str, Any], reference_df: pd.DataFrame) -> None:
    if feature in df.columns:
        value = float(reference_df[feature].quantile(0.75))
        clipped = numeric_clip(feature, value, artifact)
        df.loc[:, feature] = np.maximum(df[feature].astype(float), clipped)


def reduce_sugary_50(df: pd.DataFrame, artifact: dict[str, Any], _: pd.DataFrame) -> pd.DataFrame:
    adjusted = df.copy()
    if "total_sugary_week" in adjusted.columns:
        adjusted.loc[:, "total_sugary_week"] = adjusted["total_sugary_week"].astype(float) * 0.5
        adjusted.loc[:, "total_sugary_week"] = adjusted["total_sugary_week"].map(
            lambda value: numeric_clip("total_sugary_week", float(value), artifact)
        )
    return adjusted


def reduce_sugary_zero(df: pd.DataFrame, artifact: dict[str, Any], _: pd.DataFrame) -> pd.DataFrame:
    adjusted = df.copy()
    set_min(adjusted, "total_sugary_week", artifact)
    return adjusted


def exercise_p75(df: pd.DataFrame, artifact: dict[str, Any], reference_df: pd.DataFrame) -> pd.DataFrame:
    adjusted = df.copy()
    set_p75(adjusted, "total_exercise_week", artifact, reference_df)
    return adjusted


def activity_p75(df: pd.DataFrame, artifact: dict[str, Any], reference_df: pd.DataFrame) -> pd.DataFrame:
    adjusted = df.copy()
    set_p75(adjusted, "total_phy_activity_week", artifact, reference_df)
    return adjusted


def veg_fruit_p75(df: pd.DataFrame, artifact: dict[str, Any], reference_df: pd.DataFrame) -> pd.DataFrame:
    adjusted = df.copy()
    set_p75(adjusted, "total_veg_fruit_week", artifact, reference_df)
    return adjusted


def bmi_minus_one(df: pd.DataFrame, artifact: dict[str, Any], _: pd.DataFrame) -> pd.DataFrame:
    adjusted = df.copy()
    if "BMI" in adjusted.columns:
        adjusted.loc[:, "BMI"] = adjusted["BMI"].astype(float) - 1.0
        adjusted.loc[:, "BMI"] = adjusted["BMI"].map(lambda value: numeric_clip("BMI", float(value), artifact))
    return adjusted


def combined_lifestyle(df: pd.DataFrame, artifact: dict[str, Any], reference_df: pd.DataFrame) -> pd.DataFrame:
    adjusted = reduce_sugary_50(df, artifact, reference_df)
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


def scenario_summary(
    artifact: dict[str, Any],
    *,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    horizon: int,
    history_years: int,
    variant: str,
) -> pd.DataFrame:
    baseline_probability = predict_probability(artifact, test_df)
    baseline_score = risk_score(baseline_probability)
    rows = []

    for preset_name, preset_fn in PRESET_REGISTRY.items():
        adjusted = preset_fn(test_df.copy(), artifact, train_df)
        adjusted = engineer_features(adjusted)
        scenario_probability = predict_probability(artifact, adjusted)
        scenario_score = risk_score(scenario_probability)
        delta_probability = scenario_probability - baseline_probability
        delta_score = scenario_score - baseline_score
        unexpected = delta_score > TOLERANCE if preset_name in FAVORABLE_PRESETS else np.zeros(len(delta_score), dtype=bool)
        no_effect = np.abs(delta_score) <= TOLERANCE
        rows.append(
            {
                "variant": variant,
                "horizon_years": horizon,
                "history_years": history_years,
                "scenario": preset_name,
                "rows": float(len(test_df)),
                "mean_delta_probability": float(delta_probability.mean()),
                "min_delta_probability": float(delta_probability.min()),
                "max_delta_probability": float(delta_probability.max()),
                "mean_delta_score": float(delta_score.mean()),
                "min_delta_score": float(delta_score.min()),
                "max_delta_score": float(delta_score.max()),
                "no_effect_rows": float(no_effect.sum()),
                "no_effect_rate": float(no_effect.mean()),
                "unexpected_increase_rows": float(unexpected.sum()),
                "unexpected_increase_rate": float(unexpected.mean()),
                "directionally_correct_rate": float(1.0 - unexpected.mean()),
            }
        )
    return pd.DataFrame(rows)


def aggregate_safety(summary: pd.DataFrame) -> pd.DataFrame:
    return (
        summary.groupby(["variant", "horizon_years", "history_years"], as_index=False)
        .agg(
            scenario_count=("scenario", "size"),
            mean_delta_score=("mean_delta_score", "mean"),
            worst_positive_delta_score=("max_delta_score", "max"),
            unexpected_increase_rows=("unexpected_increase_rows", "sum"),
            total_direction_checks=("rows", "sum"),
        )
        .assign(
            unexpected_increase_rate=lambda df: df["unexpected_increase_rows"] / df["total_direction_checks"].clip(lower=1.0),
            directionally_correct_rate=lambda df: 1.0 - df["unexpected_increase_rate"],
        )
    )


def best_test_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    test = metrics[metrics["split"].eq("test")].copy()
    return (
        test.sort_values(["variant", "horizon_years", "pr_auc", "roc_auc"], ascending=[True, True, False, False])
        .groupby(["variant", "horizon_years"], as_index=False)
        .head(1)
        .reset_index(drop=True)
    )


def leaderboard_comparison(metrics: pd.DataFrame) -> pd.DataFrame:
    ranking = pd.read_csv(PHASE4_RANKING_PATH)
    current_best = (
        ranking.sort_values(["horizon_years", "pr_auc", "roc_auc"], ascending=[True, False, False])
        .groupby("horizon_years", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    best = best_test_rows(metrics)
    mono = best[best["variant"].eq("monotonic")].rename(
        columns={
            "model_key": "monotonic_model_key",
            "model_name": "monotonic_model_name",
            "history_years": "monotonic_history_years",
            "roc_auc": "monotonic_roc_auc",
            "pr_auc": "monotonic_pr_auc",
            "brier": "monotonic_brier",
            "recall": "monotonic_recall",
            "precision": "monotonic_precision",
        }
    )
    unconstrained = best[best["variant"].eq("unconstrained")].rename(
        columns={
            "model_key": "unconstrained_model_key",
            "model_name": "unconstrained_model_name",
            "history_years": "unconstrained_history_years",
            "roc_auc": "unconstrained_roc_auc",
            "pr_auc": "unconstrained_pr_auc",
            "brier": "unconstrained_brier",
            "recall": "unconstrained_recall",
            "precision": "unconstrained_precision",
        }
    )

    comparison = current_best.merge(
        mono[
            [
                "horizon_years",
                "monotonic_model_key",
                "monotonic_model_name",
                "monotonic_history_years",
                "monotonic_roc_auc",
                "monotonic_pr_auc",
                "monotonic_brier",
                "monotonic_recall",
                "monotonic_precision",
            ]
        ],
        on="horizon_years",
        how="left",
    ).merge(
        unconstrained[
            [
                "horizon_years",
                "unconstrained_model_key",
                "unconstrained_model_name",
                "unconstrained_history_years",
                "unconstrained_roc_auc",
                "unconstrained_pr_auc",
                "unconstrained_brier",
                "unconstrained_recall",
                "unconstrained_precision",
            ]
        ],
        on="horizon_years",
        how="left",
    )

    comparison = comparison.rename(
        columns={
            "model_key": "leader_model_key",
            "model_name": "leader_model_name",
            "history_years": "leader_history_years",
            "roc_auc": "leader_roc_auc",
            "pr_auc": "leader_pr_auc",
            "brier": "leader_brier",
        }
    )
    comparison["monotonic_minus_unconstrained_pr_auc"] = comparison["monotonic_pr_auc"] - comparison["unconstrained_pr_auc"]
    comparison["monotonic_minus_leader_pr_auc"] = comparison["monotonic_pr_auc"] - comparison["leader_pr_auc"]
    comparison["monotonic_minus_unconstrained_roc_auc"] = comparison["monotonic_roc_auc"] - comparison["unconstrained_roc_auc"]
    comparison["monotonic_minus_leader_roc_auc"] = comparison["monotonic_roc_auc"] - comparison["leader_roc_auc"]
    comparison["monotonic_minus_unconstrained_recall"] = comparison["monotonic_recall"] - comparison["unconstrained_recall"]
    return comparison


def select_best_safety_rows(
    comparison: pd.DataFrame,
    scenario_summary_df: pd.DataFrame,
    safety_summary: pd.DataFrame,
) -> pd.DataFrame:
    del scenario_summary_df
    chosen = []
    for _, row in comparison.iterrows():
        horizon = int(row["horizon_years"])
        chosen.append(
            {
                "variant": "monotonic",
                "horizon_years": horizon,
                "history_years": int(row["monotonic_history_years"]),
            }
        )
        chosen.append(
            {
                "variant": "unconstrained",
                "horizon_years": horizon,
                "history_years": int(row["unconstrained_history_years"]),
            }
        )
    chooser = pd.DataFrame(chosen).drop_duplicates()
    return (
        safety_summary.merge(chooser, on=["variant", "horizon_years", "history_years"], how="inner")
        .sort_values(["horizon_years", "variant"])
        .reset_index(drop=True)
    )


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
