"""Phase 6 v2 monotonic-vs-unconstrained LightGBM ablation.

This mirrors the Phase 6 XGBoost ablation so intervention-ready LightGBM can be
compared against:
  - the current final pure-prediction leaderboard
  - the current monotonic XGBoost intervention benchmark

Run from the repository root:
    python digihealth_risk/phase_5/train_monotonic_lightgbm.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from lightgbm import LGBMClassifier


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from digihealth_risk.phase_2.train_tree_models import (  # noqa: E402
    RANDOM_SEED,
    classification_metrics,
    engineer_features,
    get_feature_columns,
    load_table,
    make_preprocessor,
    split_by_patient,
)
from digihealth_risk.phase_5.monotonic_ablation_utils import (  # noqa: E402
    HISTORY_OPTIONS,
    HORIZONS,
    OUT_DIR,
    PHASE4_RANKING_PATH,
    PRESET_REGISTRY,
    aggregate_safety,
    best_test_rows,
    constraint_table,
    leaderboard_comparison,
    markdown_table,
    monotone_constraints,
    phase0_path,
    risk_score,
    scenario_summary,
    select_best_safety_rows,
)

OUT_DIR = ROOT / "digihealth_risk" / "phase_5" / "outputs"
PHASE4_RANKING_PATH = ROOT / "digihealth_risk" / "phase_4" / "outputs" / "phase_4_2_v2_cross_family_ranking.csv"


def phase0_path(horizon: int, history_years: int) -> Path:
    p = OUT_DIR.parents[1] / "phase_0" / "outputs" / f"phase_0_modeling_table_horizon_{horizon}_history_{history_years}.pkl"
    if p.exists():
        return p
    d = OUT_DIR.parents[1] / "phase_0" / "outputs" / "phase_0_modeling_table.pkl"
    if horizon == 1 and history_years == 1 and d.exists():
        return d
    raise FileNotFoundError(f"No Phase 0 table for horizon={horizon}, history={history_years}.")


MODEL_DIR = OUT_DIR / "models_v2_lightgbm_ablation"
XGB_ABLATION_METRICS_PATH = OUT_DIR / "phase_6_v2_ablation_metrics.csv"
XGB_ABLATION_SAFETY_PATH = OUT_DIR / "phase_6_v2_ablation_safety_summary.csv"
VARIANTS = ["unconstrained", "monotonic"]


def model_key(variant: str, horizon: int, history_years: int) -> str:
    return f"phase6_v2_{variant}_lightgbm_n{horizon}_m{history_years}"


def model_name(variant: str) -> str:
    return f"{variant}_lightgbm_v2"


def build_lightgbm(*, constraints: tuple[int, ...] | None) -> LGBMClassifier:
    kwargs: dict[str, Any] = {
        "n_estimators": 400,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "min_child_samples": 40,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 2.0,
        "objective": "binary",
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
        "verbosity": -1,
    }
    if constraints is not None:
        kwargs["monotone_constraints"] = list(constraints)
        kwargs["monotone_constraints_method"] = "advanced"
    return LGBMClassifier(**kwargs)


def fit_variant(train_df: pd.DataFrame, *, variant: str, history_years: int) -> dict[str, Any]:
    numeric_features, categorical_features = get_feature_columns(train_df)
    feature_columns = numeric_features + categorical_features
    preprocessor = make_preprocessor(numeric_features, categorical_features)
    x_train = preprocessor.fit_transform(train_df[feature_columns].copy())
    transformed_names = [str(name) for name in preprocessor.get_feature_names_out()]
    constraints = monotone_constraints(transformed_names, history_years) if variant == "monotonic" else tuple([0] * len(transformed_names))
    model = build_lightgbm(constraints=constraints if variant == "monotonic" else None)
    y_train = train_df["Target_AtRisk_Status"].astype(int).to_numpy()
    model.fit(x_train, y_train)
    return {
        "variant": variant,
        "preprocessor": preprocessor,
        "model": model,
        "feature_columns": feature_columns,
        "transformed_feature_names": transformed_names,
        "monotone_constraints": constraints,
    }


def predict_probability(artifact: dict[str, Any], df: pd.DataFrame):
    x = artifact["preprocessor"].transform(df[artifact["feature_columns"]].copy())
    return artifact["model"].predict_proba(x)[:, 1]


def run_combo(horizon: int, history_years: int, variant: str):
    df = load_table(phase0_path(horizon, history_years))
    df = engineer_features(df)
    train_df, test_df = split_by_patient(df)
    artifact = fit_variant(train_df, variant=variant, history_years=history_years)

    train_probability = predict_probability(artifact, train_df)
    test_probability = predict_probability(artifact, test_df)
    y_train = train_df["Target_AtRisk_Status"].astype(int).to_numpy()
    y_test = test_df["Target_AtRisk_Status"].astype(int).to_numpy()
    threshold = float(y_train.mean())

    metrics = pd.DataFrame(
        [
            {
                "variant": variant,
                "horizon_years": horizon,
                "history_years": history_years,
                "model_key": model_key(variant, horizon, history_years),
                "model_name": model_name(variant),
                "split": "train",
                **classification_metrics(y_train, train_probability, threshold),
            },
            {
                "variant": variant,
                "horizon_years": horizon,
                "history_years": history_years,
                "model_key": model_key(variant, horizon, history_years),
                "model_name": model_name(variant),
                "split": "test",
                **classification_metrics(y_test, test_probability, threshold),
            },
        ]
    )

    predictions = test_df[["PatientId", "Year", "target_year", "Target_AtRisk_Status"]].copy()
    predictions["variant"] = variant
    predictions["horizon_years"] = horizon
    predictions["history_years"] = history_years
    predictions["model_key"] = model_key(variant, horizon, history_years)
    predictions["model_name"] = model_name(variant)
    predictions["predicted_probability"] = test_probability
    predictions["risk_score_0_100"] = risk_score(test_probability)

    constraints = constraint_table(
        artifact["transformed_feature_names"],
        artifact["monotone_constraints"],
        horizon=horizon,
        history_years=history_years,
        variant=variant,
    )

    artifact.update(
        {
            "horizon_years": horizon,
            "history_years": history_years,
            "model_key": model_key(variant, horizon, history_years),
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
    joblib.dump(artifact, MODEL_DIR / f"{model_key(variant, horizon, history_years)}.joblib")

    scenarios = scenario_summary(
        artifact,
        train_df=train_df,
        test_df=test_df,
        horizon=horizon,
        history_years=history_years,
        variant=variant,
    )
    return metrics, predictions, constraints, scenarios


def compare_against_xgb(metrics: pd.DataFrame) -> pd.DataFrame:
    xgb = pd.read_csv(XGB_ABLATION_METRICS_PATH)
    xgb_best = (
        best_test_rows(xgb)
        .query("variant == 'monotonic'")
        .rename(
            columns={
                "model_key": "xgb_model_key",
                "model_name": "xgb_model_name",
                "history_years": "xgb_history_years",
                "roc_auc": "xgb_roc_auc",
                "pr_auc": "xgb_pr_auc",
                "brier": "xgb_brier",
                "recall": "xgb_recall",
                "precision": "xgb_precision",
            }
        )
    )
    lgbm_best = (
        best_test_rows(metrics)
        .query("variant == 'monotonic'")
        .rename(
            columns={
                "model_key": "lgbm_model_key",
                "model_name": "lgbm_model_name",
                "history_years": "lgbm_history_years",
                "roc_auc": "lgbm_roc_auc",
                "pr_auc": "lgbm_pr_auc",
                "brier": "lgbm_brier",
                "recall": "lgbm_recall",
                "precision": "lgbm_precision",
            }
        )
    )
    comparison = lgbm_best.merge(
        xgb_best[
            [
                "horizon_years",
                "xgb_model_key",
                "xgb_model_name",
                "xgb_history_years",
                "xgb_roc_auc",
                "xgb_pr_auc",
                "xgb_brier",
                "xgb_recall",
                "xgb_precision",
            ]
        ],
        on="horizon_years",
        how="left",
    )
    comparison["lgbm_minus_xgb_pr_auc"] = comparison["lgbm_pr_auc"] - comparison["xgb_pr_auc"]
    comparison["lgbm_minus_xgb_roc_auc"] = comparison["lgbm_roc_auc"] - comparison["xgb_roc_auc"]
    comparison["lgbm_minus_xgb_recall"] = comparison["lgbm_recall"] - comparison["xgb_recall"]
    return comparison


def compare_safety_against_xgb(safety_summary_df: pd.DataFrame, comparison_df: pd.DataFrame) -> pd.DataFrame:
    xgb = pd.read_csv(XGB_ABLATION_SAFETY_PATH)
    chosen = comparison_df[["horizon_years", "lgbm_history_years", "xgb_history_years"]].copy()
    lgbm = safety_summary_df.merge(
        chosen.rename(columns={"lgbm_history_years": "history_years"}),
        on=["horizon_years", "history_years"],
        how="inner",
    )
    lgbm = lgbm[lgbm["variant"].eq("monotonic")].rename(
        columns={
            "history_years": "lgbm_history_years",
            "directionally_correct_rate": "lgbm_directionally_correct_rate",
            "unexpected_increase_rate": "lgbm_unexpected_increase_rate",
            "mean_delta_score": "lgbm_mean_delta_score",
            "worst_positive_delta_score": "lgbm_worst_positive_delta_score",
        }
    )
    xgb = xgb[xgb["variant"].eq("monotonic")].rename(
        columns={
            "history_years": "xgb_history_years",
            "directionally_correct_rate": "xgb_directionally_correct_rate",
            "unexpected_increase_rate": "xgb_unexpected_increase_rate",
            "mean_delta_score": "xgb_mean_delta_score",
            "worst_positive_delta_score": "xgb_worst_positive_delta_score",
        }
    )
    result = lgbm.merge(
        xgb[
            [
                "horizon_years",
                "xgb_history_years",
                "xgb_directionally_correct_rate",
                "xgb_unexpected_increase_rate",
                "xgb_mean_delta_score",
                "xgb_worst_positive_delta_score",
            ]
        ],
        on=["horizon_years", "xgb_history_years"],
        how="left",
    )
    result["lgbm_minus_xgb_directionally_correct_rate"] = (
        result["lgbm_directionally_correct_rate"] - result["xgb_directionally_correct_rate"]
    )
    return result


def write_report(
    metrics: pd.DataFrame,
    leaderboard_comparison_df: pd.DataFrame,
    xgb_comparison_df: pd.DataFrame,
    best_safety_df: pd.DataFrame,
    xgb_safety_df: pd.DataFrame,
) -> str:
    best_rows = best_test_rows(metrics)
    metric_cols = [
        "variant",
        "horizon_years",
        "history_years",
        "pr_auc",
        "roc_auc",
        "brier",
        "precision",
        "recall",
        "specificity",
        "f1",
    ]
    lines = [
        "# Phase 6 v2 Monotonic LightGBM Ablation Report",
        "",
        "## Scope",
        "Benchmarks monotonic LightGBM against an otherwise identical unconstrained LightGBM. "
        "Results are compared against both the current final pure-prediction leaderboard and "
        "the current monotonic XGBoost intervention benchmark.",
        "",
        "## Best Test Metrics Per Variant/Horizon",
        markdown_table(best_rows[metric_cols]),
        "",
        "## Comparison Against Final Leaderboard",
        markdown_table(
            leaderboard_comparison_df[
                [
                    "horizon_years",
                    "leader_model_key",
                    "leader_pr_auc",
                    "unconstrained_model_key",
                    "unconstrained_history_years",
                    "unconstrained_pr_auc",
                    "monotonic_model_key",
                    "monotonic_history_years",
                    "monotonic_pr_auc",
                    "monotonic_minus_unconstrained_pr_auc",
                    "monotonic_minus_leader_pr_auc",
                ]
            ]
        ),
        "",
        "## Comparison Against Monotonic XGBoost",
        markdown_table(
            xgb_comparison_df[
                [
                    "horizon_years",
                    "xgb_model_key",
                    "xgb_pr_auc",
                    "lgbm_model_key",
                    "lgbm_pr_auc",
                    "lgbm_minus_xgb_pr_auc",
                ]
            ]
        ),
        "",
        "## LightGBM Intervention Safety Summary",
        markdown_table(
            best_safety_df[
                [
                    "variant",
                    "horizon_years",
                    "history_years",
                    "scenario_count",
                    "directionally_correct_rate",
                    "unexpected_increase_rate",
                    "mean_delta_score",
                    "worst_positive_delta_score",
                ]
            ]
        ),
        "",
        "## Monotonic LightGBM vs Monotonic XGBoost Safety",
        markdown_table(
            xgb_safety_df[
                [
                    "horizon_years",
                    "xgb_history_years",
                    "xgb_directionally_correct_rate",
                    "lgbm_history_years",
                    "lgbm_directionally_correct_rate",
                    "lgbm_minus_xgb_directionally_correct_rate",
                    "xgb_unexpected_increase_rate",
                    "lgbm_unexpected_increase_rate",
                ]
            ]
        ),
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    metric_parts = []
    prediction_parts = []
    constraint_parts = []
    scenario_parts = []

    for history_years in HISTORY_OPTIONS:
        for horizon in HORIZONS:
            for variant in VARIANTS:
                print(f"Training {variant} LightGBM v2 N={horizon}, M={history_years}")
                metrics, predictions, constraints, scenarios = run_combo(horizon, history_years, variant)
                metric_parts.append(metrics)
                prediction_parts.append(predictions)
                constraint_parts.append(constraints)
                scenario_parts.append(scenarios)

    metrics_df = pd.concat(metric_parts, ignore_index=True)
    predictions_df = pd.concat(prediction_parts, ignore_index=True)
    constraints_df = pd.concat(constraint_parts, ignore_index=True)
    scenario_summary_df = pd.concat(scenario_parts, ignore_index=True)
    safety_summary_df = aggregate_safety(scenario_summary_df)
    leaderboard_comparison_df = leaderboard_comparison(metrics_df)
    xgb_comparison_df = compare_against_xgb(metrics_df)
    best_safety_df = select_best_safety_rows(leaderboard_comparison_df, scenario_summary_df, safety_summary_df)
    xgb_safety_df = compare_safety_against_xgb(safety_summary_df, xgb_comparison_df)

    metrics_df.to_csv(OUT_DIR / "phase_6_v2_lightgbm_ablation_metrics.csv", index=False)
    predictions_df.to_csv(OUT_DIR / "phase_6_v2_lightgbm_ablation_test_predictions.csv", index=False)
    constraints_df.to_csv(OUT_DIR / "phase_6_v2_lightgbm_ablation_constraints.csv", index=False)
    scenario_summary_df.to_csv(OUT_DIR / "phase_6_v2_lightgbm_ablation_scenario_summary.csv", index=False)
    safety_summary_df.to_csv(OUT_DIR / "phase_6_v2_lightgbm_ablation_safety_summary.csv", index=False)
    leaderboard_comparison_df.to_csv(OUT_DIR / "phase_6_v2_lightgbm_ablation_vs_leaderboard.csv", index=False)
    xgb_comparison_df.to_csv(OUT_DIR / "phase_6_v2_lightgbm_ablation_vs_monotonic_xgboost.csv", index=False)

    report = write_report(
        metrics_df,
        leaderboard_comparison_df,
        xgb_comparison_df,
        best_safety_df,
        xgb_safety_df,
    )
    (OUT_DIR / "phase_6_v2_lightgbm_ablation_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
