"""Phase 1 random-intercept logistic GLMM with statsmodels.

Fits a Bayesian binomial mixed model using patient-level random intercepts:

    logit(P(next-year at-risk)) = fixed effects + (1 | PatientId)

Run from the repository root:
    python digihealth_risk/phase_1/glmm_exploratory.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from patsy import build_design_matrices
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from digihealth_risk.utils.patient_split import apply_canonical_split  # noqa: E402

INPUT_PATH = ROOT / "digihealth_risk" / "phase_0" / "outputs" / "phase_0_modeling_table.pkl"
OUT_DIR = ROOT / "digihealth_risk" / "phase_1" / "outputs"
RANDOM_SEED = 20260501

CONTINUOUS_FEATURES = [
    "Year_centered",
    "Age",
    "FBS",
    "BMI",
    "Pulse",
    "BL_pres1",
    "Waist",
    "pulse_pressure",
    "MAX_FBS_up_to_year",
    "years_since_last_fbs",
    "clinical_observed_count",
    "total_sugary_week",
    "total_veg_fruit_week",
    "total_exercise_week",
    "total_phy_activity_week",
    "sleep_hours",
]

CATEGORICAL_FEATURES = [
    "gender",
    "dm_first_degree_relative",
    "sleep_quality",
    "smoking_status",
    "alcohol_status",
]

MISSING_INDICATOR_FEATURES = [
    "FBS",
    "BMI",
    "Pulse",
    "BL_pres1",
    "Waist",
    "pulse_pressure",
    "years_since_last_fbs",
]


def install_numpy_pickle_compat() -> None:
    """Allow NumPy 1.x to read pickles created by NumPy 2.x."""
    import sys
    import numpy.core as np_core
    import numpy.core.multiarray as np_multiarray
    import numpy.core.numeric as np_numeric

    sys.modules.setdefault("numpy._core", np_core)
    sys.modules.setdefault("numpy._core.multiarray", np_multiarray)
    sys.modules.setdefault("numpy._core.numeric", np_numeric)


def load_data() -> pd.DataFrame:
    install_numpy_pickle_compat()
    df = pd.read_pickle(INPUT_PATH).copy()
    if "Target_AtRisk_Status" not in df.columns:
        df["Target_AtRisk_Status"] = df["Next_Year_AtRisk_Status"]
    df["Year_centered"] = df["Year"] - df["Year"].min()
    return df


def split_by_patient(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return apply_canonical_split(df)


def prepare_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train = train_df.copy()
    test = test_df.copy()
    z_features: list[str] = []

    for feature in CONTINUOUS_FEATURES:
        median = train[feature].median()
        train_filled = train[feature].fillna(median)
        mean = train_filled.mean()
        std = train_filled.std(ddof=0)
        if not np.isfinite(std) or std == 0:
            std = 1.0

        z_feature = f"{feature}_z"
        train[z_feature] = (train_filled - mean) / std
        test[z_feature] = (test[feature].fillna(median) - mean) / std
        z_features.append(z_feature)

    for feature in MISSING_INDICATOR_FEATURES:
        missing_feature = f"{feature}_missing"
        train[missing_feature] = train[feature].isna().astype(int)
        test[missing_feature] = test[feature].isna().astype(int)

    for feature in CATEGORICAL_FEATURES:
        train[feature] = train[feature].astype("object").fillna("missing").astype(str)
        test[feature] = test[feature].astype("object").fillna("missing").astype(str)

    train["is_missing_last_year_filled"] = train["is_missing_last_year"].fillna(False).astype(int)
    test["is_missing_last_year_filled"] = test["is_missing_last_year"].fillna(False).astype(int)
    train["has_fbs_this_year"] = train["has_fbs_this_year"].astype(int)
    test["has_fbs_this_year"] = test["has_fbs_this_year"].astype(int)
    train["target"] = train["Target_AtRisk_Status"].astype(int)
    test["target"] = test["Target_AtRisk_Status"].astype(int)

    return train, test, z_features


def build_formula(z_features: list[str]) -> str:
    fixed_terms = (
        z_features
        + [f"{feature}_missing" for feature in MISSING_INDICATOR_FEATURES]
        + ["has_fbs_this_year", "is_missing_last_year_filled"]
        + [f"C({feature})" for feature in CATEGORICAL_FEATURES]
    )
    return "target ~ " + " + ".join(fixed_terms)


def classification_metrics(y_true: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float]:
    pred = probability >= threshold
    return {
        "rows": float(len(y_true)),
        "positives": float(y_true.sum()),
        "positive_rate": float(y_true.mean()),
        "threshold": threshold,
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "pr_auc": float(average_precision_score(y_true, probability)),
        "brier": float(brier_score_loss(y_true, probability)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "specificity": float(((~pred) & (y_true == 0)).sum() / max((y_true == 0).sum(), 1)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
    }


def coefficients_table(result: object, exog_names: list[str]) -> pd.DataFrame:
    table = pd.DataFrame(
        {
            "feature": exog_names,
            "posterior_mean": result.fe_mean,
            "posterior_sd": result.fe_sd,
        }
    )
    table["odds_ratio"] = np.exp(table["posterior_mean"])
    table["or_ci_low"] = np.exp(table["posterior_mean"] - 1.96 * table["posterior_sd"])
    table["or_ci_high"] = np.exp(table["posterior_mean"] + 1.96 * table["posterior_sd"])
    table["abs_log_or"] = table["posterior_mean"].abs()
    return table.sort_values("abs_log_or", ascending=False).drop(columns="abs_log_or")


def calibration_table(y_true: np.ndarray, probability: np.ndarray, bins: int = 10) -> pd.DataFrame:
    quantiles = pd.qcut(probability, q=bins, duplicates="drop")
    df = pd.DataFrame({"y": y_true, "p": probability, "bin": quantiles})
    return (
        df.groupby("bin", observed=True)
        .agg(rows=("y", "size"), mean_probability=("p", "mean"), observed_rate=("y", "mean"))
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


def write_report(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    formula: str,
    metrics_df: pd.DataFrame,
    coef_df: pd.DataFrame,
    cal_df: pd.DataFrame,
    random_sd: float,
) -> str:
    lines = [
        "# Phase 1 Random-Intercept Logistic GLMM",
        "",
        "## Scope",
        "This model uses `statsmodels.genmod.bayes_mixed_glm.BinomialBayesMixedGLM` "
        "to fit a binomial mixed model with a patient-level random intercept.",
        "",
        "## Data Split",
        f"Grouped by `PatientId`: `{train_df['PatientId'].nunique():,}` train patients, "
        f"`{test_df['PatientId'].nunique():,}` test patients.",
        "",
        "## Formula",
        f"`{formula}`",
        "",
        "Variance component: `(1 | PatientId)`. "
        f"Estimated patient random-intercept SD: `{random_sd:.4f}`.",
        "",
        "## Metrics",
        markdown_table(metrics_df),
        "",
        "## Most Influential Fixed Effects",
        "Continuous features are median-imputed and standardized on the training set. "
        "Test predictions use fixed effects only for unseen patients.",
        markdown_table(coef_df[coef_df["feature"] != "Intercept"], max_rows=30),
        "",
        "## Test Calibration Deciles",
        markdown_table(cal_df),
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    raw_train, raw_test = split_by_patient(df)
    train_df, test_df, z_features = prepare_features(raw_train, raw_test)
    formula = build_formula(z_features)

    model = BinomialBayesMixedGLM.from_formula(
        formula,
        {"patient": "0 + C(PatientId)"},
        train_df,
        vcp_p=0.5,
        fe_p=2.0,
    )
    result = model.fit_vb(minim_opts={"maxiter": 500}, verbose=False)

    train_probability = result.predict()
    test_exog = build_design_matrices([model.data.design_info], test_df, return_type="dataframe")[0]
    test_probability = result.predict(exog=test_exog)

    y_train = train_df["target"].to_numpy()
    y_test = test_df["target"].to_numpy()
    threshold = float(y_train.mean())
    metrics_df = pd.DataFrame(
        [
            {"split": "train", **classification_metrics(y_train, train_probability, threshold)},
            {"split": "test", **classification_metrics(y_test, test_probability, threshold)},
        ]
    )
    coef_df = coefficients_table(result, model.exog_names)
    cal_df = calibration_table(y_test, test_probability)

    predictions = test_df[["PatientId", "Year", "target_year", "target"]].copy()
    predictions = predictions.rename(columns={"target": "Target_AtRisk_Status"})
    predictions["predicted_probability"] = test_probability

    random_sd = float(np.exp(result.vcp_mean[0]))
    coef_df.to_csv(OUT_DIR / "phase_1_glmm_fixed_effects.csv", index=False)
    metrics_df.to_csv(OUT_DIR / "phase_1_glmm_metrics.csv", index=False)
    predictions.to_csv(OUT_DIR / "phase_1_glmm_test_predictions.csv", index=False)
    cal_df.to_csv(OUT_DIR / "phase_1_glmm_test_calibration.csv", index=False)
    (OUT_DIR / "phase_1_glmm_model_metadata.json").write_text(
        json.dumps(
            {
                "formula": formula,
                "random_effect": "0 + C(PatientId)",
                "random_intercept_sd": random_sd,
                "random_seed": RANDOM_SEED,
                "test_patient_fraction": 0.2,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    report = write_report(train_df, test_df, formula, metrics_df, coef_df, cal_df, random_sd)
    (OUT_DIR / "phase_1_glmm_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
