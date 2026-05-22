"""Phase 6 v2 monotonic-vs-unconstrained logistic ablation.

This benchmark adds a simple statistical intervention baseline. It compares:
  - unconstrained_logistic_v2
  - monotonic_logistic_v2

The monotonic variant enforces coefficient sign constraints after a positive
feature scaling transform so favorable interventions cannot reverse direction
through the constrained features.

Run from the repository root:
    python digihealth_risk/phase_5/train_monotonic_logistic.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import optimize, special
from sklearn.linear_model import LogisticRegression


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
    FAVORABLE_PRESETS,
    HISTORY_OPTIONS,
    HORIZONS,
    OUT_DIR,
    PRESET_REGISTRY,
    TOLERANCE,
    aggregate_safety,
    best_test_rows,
    constraint_table,
    leaderboard_comparison,
    markdown_table,
    monotone_constraints,
    phase0_path,
    risk_score,
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


RIDGE_ALPHA = 0.01
MODEL_DIR = OUT_DIR / "models_v2_logistic_ablation"
XGB_ABLATION_METRICS_PATH = OUT_DIR / "phase_6_v2_ablation_metrics.csv"
XGB_ABLATION_SAFETY_PATH = OUT_DIR / "phase_6_v2_ablation_safety_summary.csv"
VARIANTS = ["unconstrained", "monotonic"]


def model_key(variant: str, horizon: int, history_years: int) -> str:
    return f"phase6_v2_{variant}_logistic_n{horizon}_m{history_years}"


def model_name(variant: str) -> str:
    return f"{variant}_logistic_v2"


def prepare_matrix(
    df: pd.DataFrame,
    preprocessor: Any,
    feature_columns: list[str],
    mean_: np.ndarray,
    scale_: np.ndarray,
) -> np.ndarray:
    x_raw = preprocessor.transform(df[feature_columns].copy()).astype(float)
    x_scaled = (x_raw - mean_) / scale_
    intercept = np.ones((x_scaled.shape[0], 1), dtype=float)
    return np.hstack([intercept, x_scaled])


def coefficient_bounds(constraints: tuple[int, ...], variant: str) -> list[tuple[float | None, float | None]]:
    bounds: list[tuple[float | None, float | None]] = [(None, None)]
    for sign in constraints:
        if variant != "monotonic" or sign == 0:
            bounds.append((None, None))
        elif sign == 1:
            bounds.append((0.0, None))
        else:
            bounds.append((None, 0.0))
    return bounds


def negative_log_likelihood(beta: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    eta = x @ beta
    nll = np.sum(np.logaddexp(0.0, eta) - y * eta)
    probability = special.expit(eta)
    gradient = x.T @ (probability - y)

    beta_penalty = beta.copy()
    beta_penalty[0] = 0.0
    nll += 0.5 * RIDGE_ALPHA * np.dot(beta_penalty, beta_penalty)
    gradient += RIDGE_ALPHA * beta_penalty
    return float(nll), gradient


def fit_variant(train_df: pd.DataFrame, *, variant: str, history_years: int) -> dict[str, Any]:
    numeric_features, categorical_features = get_feature_columns(train_df)
    feature_columns = numeric_features + categorical_features
    preprocessor = make_preprocessor(numeric_features, categorical_features)
    x_raw = preprocessor.fit_transform(train_df[feature_columns].copy()).astype(float)
    transformed_names = [str(name) for name in preprocessor.get_feature_names_out()]
    constraints = (
        monotone_constraints(transformed_names, history_years)
        if variant == "monotonic"
        else tuple([0] * len(transformed_names))
    )

    mean_ = x_raw.mean(axis=0)
    scale_ = x_raw.std(axis=0)
    scale_[scale_ == 0.0] = 1.0
    x_scaled = (x_raw - mean_) / scale_
    x_train = np.hstack([np.ones((x_scaled.shape[0], 1), dtype=float), x_scaled])
    y_train = train_df["Target_AtRisk_Status"].astype(float).to_numpy()

    base_model = LogisticRegression(
        C=max(1.0 / RIDGE_ALPHA, 1e-6),
        solver="lbfgs",
        fit_intercept=True,
        max_iter=2000,
        random_state=RANDOM_SEED,
    )
    base_model.fit(x_scaled, y_train.astype(int))
    initial = np.concatenate(
        [
            np.atleast_1d(base_model.intercept_).astype(float),
            np.ravel(base_model.coef_).astype(float),
        ]
    )

    if variant == "unconstrained":
        coefficients = initial
    else:
        bounded = coefficient_bounds(constraints, variant)
        clipped = [initial[0]]
        for value, bounds in zip(initial[1:], bounded[1:], strict=True):
            lower, upper = bounds
            clipped.append(float(np.clip(value, lower if lower is not None else value, upper if upper is not None else value)))
        initial = np.asarray(clipped, dtype=float)
        result = optimize.minimize(
            fun=lambda beta: negative_log_likelihood(beta, x_train, y_train),
            x0=initial,
            jac=True,
            method="L-BFGS-B",
            bounds=coefficient_bounds(constraints, variant),
            options={"maxiter": 5000, "ftol": 1e-8, "maxls": 100},
        )
        if not result.success:
            raise RuntimeError(f"Logistic optimization failed: {result.message}")
        coefficients = result.x

    return {
        "variant": variant,
        "preprocessor": preprocessor,
        "feature_columns": feature_columns,
        "transformed_feature_names": transformed_names,
        "monotone_constraints": constraints,
        "mean_": mean_,
        "scale_": scale_,
        "coefficients": coefficients,
    }


def predict_probability(artifact: dict[str, Any], df: pd.DataFrame) -> np.ndarray:
    x = prepare_matrix(
        df,
        artifact["preprocessor"],
        artifact["feature_columns"],
        artifact["mean_"],
        artifact["scale_"],
    )
    return special.expit(x @ artifact["coefficients"])


def scenario_summary_logistic(
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
                    "min": float(train_df[column].min(skipna=True))
                    if pd.api.types.is_numeric_dtype(train_df[column])
                    else None,
                    "max": float(train_df[column].max(skipna=True))
                    if pd.api.types.is_numeric_dtype(train_df[column])
                    else None,
                }
                for column in artifact["feature_columns"]
            },
        }
    )
    joblib.dump(artifact, MODEL_DIR / f"{model_key(variant, horizon, history_years)}.joblib")

    scenarios = scenario_summary_logistic(
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
    log_best = (
        best_test_rows(metrics)
        .query("variant == 'monotonic'")
        .rename(
            columns={
                "model_key": "log_model_key",
                "model_name": "log_model_name",
                "history_years": "log_history_years",
                "roc_auc": "log_roc_auc",
                "pr_auc": "log_pr_auc",
                "brier": "log_brier",
                "recall": "log_recall",
                "precision": "log_precision",
            }
        )
    )
    comparison = log_best.merge(
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
    comparison["log_minus_xgb_pr_auc"] = comparison["log_pr_auc"] - comparison["xgb_pr_auc"]
    comparison["log_minus_xgb_roc_auc"] = comparison["log_roc_auc"] - comparison["xgb_roc_auc"]
    comparison["log_minus_xgb_recall"] = comparison["log_recall"] - comparison["xgb_recall"]
    return comparison


def compare_safety_against_xgb(safety_summary_df: pd.DataFrame, comparison_df: pd.DataFrame) -> pd.DataFrame:
    xgb = pd.read_csv(XGB_ABLATION_SAFETY_PATH)
    chosen = comparison_df[["horizon_years", "log_history_years", "xgb_history_years"]].copy()
    log_df = safety_summary_df.merge(
        chosen.rename(columns={"log_history_years": "history_years"}),
        on=["horizon_years", "history_years"],
        how="inner",
    )
    log_df = log_df[log_df["variant"].eq("monotonic")].rename(
        columns={
            "history_years": "log_history_years",
            "directionally_correct_rate": "log_directionally_correct_rate",
            "unexpected_increase_rate": "log_unexpected_increase_rate",
            "mean_delta_score": "log_mean_delta_score",
            "worst_positive_delta_score": "log_worst_positive_delta_score",
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
    result = log_df.merge(
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
    result["log_minus_xgb_directionally_correct_rate"] = (
        result["log_directionally_correct_rate"] - result["xgb_directionally_correct_rate"]
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
        "# Phase 6 v2 Monotonic Logistic Ablation Report",
        "",
        "## Scope",
        "Benchmarks a monotonic logistic baseline against an otherwise identical unconstrained logistic model. "
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
                    "log_model_key",
                    "log_pr_auc",
                    "log_minus_xgb_pr_auc",
                ]
            ]
        ),
        "",
        "## Logistic Intervention Safety Summary",
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
        "## Monotonic Logistic vs Monotonic XGBoost Safety",
        markdown_table(
            xgb_safety_df[
                [
                    "horizon_years",
                    "xgb_history_years",
                    "xgb_directionally_correct_rate",
                    "log_history_years",
                    "log_directionally_correct_rate",
                    "log_minus_xgb_directionally_correct_rate",
                    "xgb_unexpected_increase_rate",
                    "log_unexpected_increase_rate",
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
                print(f"Training {variant} logistic v2 N={horizon}, M={history_years}")
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
    best_safety_df = select_best_safety_rows(
        leaderboard_comparison_df, scenario_summary_df, safety_summary_df
    )
    xgb_safety_df = compare_safety_against_xgb(safety_summary_df, xgb_comparison_df)

    metrics_df.to_csv(OUT_DIR / "phase_6_v2_logistic_ablation_metrics.csv", index=False)
    predictions_df.to_csv(
        OUT_DIR / "phase_6_v2_logistic_ablation_test_predictions.csv", index=False
    )
    constraints_df.to_csv(
        OUT_DIR / "phase_6_v2_logistic_ablation_constraints.csv", index=False
    )
    scenario_summary_df.to_csv(
        OUT_DIR / "phase_6_v2_logistic_ablation_scenario_summary.csv", index=False
    )
    safety_summary_df.to_csv(
        OUT_DIR / "phase_6_v2_logistic_ablation_safety_summary.csv", index=False
    )
    leaderboard_comparison_df.to_csv(
        OUT_DIR / "phase_6_v2_logistic_ablation_vs_leaderboard.csv", index=False
    )
    xgb_comparison_df.to_csv(
        OUT_DIR / "phase_6_v2_logistic_ablation_vs_monotonic_xgboost.csv", index=False
    )

    report = write_report(
        metrics_df,
        leaderboard_comparison_df,
        xgb_comparison_df,
        best_safety_df,
        xgb_safety_df,
    )
    (OUT_DIR / "phase_6_v2_logistic_ablation_report.md").write_text(
        report, encoding="utf-8"
    )
    print(report)


if __name__ == "__main__":
    main()
