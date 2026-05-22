"""Phase 1 v2 GEE logistic model — improved feature engineering.

Changes from v1 (phase_1_gee_statsmodels.py) driven by Phase 0.2 EDA:
  - Removed `pulse_pressure` (VIF=225; = BL_pres1 − BL_pres2 by construction)
  - Added FBS hockey-stick features at clinical thresholds (100 mg/dL pre-DM,
    125 mg/dL DM) to capture non-linear risk structure
  - Added `Year_centered_sq` for the non-linear temporal trend (Ljung-Box p=0.03)
  - Added `FBS_x_Age` interaction (already in logistic v1; top two predictors)

Run from the repository root:
    python digihealth_risk/phase_1/gee.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from digihealth_risk.utils.patient_split import apply_canonical_split  # noqa: E402

INPUT_PATH = ROOT / "digihealth_risk" / "phase_0" / "outputs" / "phase_0_modeling_table.pkl"
OUT_DIR = ROOT / "digihealth_risk" / "phase_1" / "outputs"
RANDOM_SEED = 20260501

BASE_CONTINUOUS_FEATURES = [
    "Year_centered",
    "Year_centered_sq",       # v2: non-linear time trend
    "Age",
    "FBS",
    "FBS_hinge_100",          # v2: hockey-stick at pre-DM threshold 100 mg/dL
    "FBS_hinge_125",          # v2: hockey-stick at DM threshold 125 mg/dL
    "BMI",
    "Pulse",
    "BL_pres1",
    "Waist",
    # pulse_pressure removed: VIF=225, = BL_pres1 - BL_pres2 by construction
    "MAX_FBS_up_to_year",
    "years_since_last_fbs",
    "clinical_observed_count",
    "FBS_x_Age",              # v2: interaction; already in logistic v1
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

# GEE has no regularization — individual clinical missing indicators cause
# extreme collinear coefficients (all flip together on unvisited years).
# Missingness is captured by clinical_observed_count_z and years_since_last_fbs_z.
MISSING_INDICATOR_FEATURES: list[str] = []

HISTORY_FEATURE_PATTERNS = ("_mean", "_slope")


def install_numpy_pickle_compat() -> None:
    import sys
    import numpy.core as np_core
    import numpy.core.multiarray as np_multiarray
    import numpy.core.numeric as np_numeric

    sys.modules.setdefault("numpy._core", np_core)
    sys.modules.setdefault("numpy._core.multiarray", np_multiarray)
    sys.modules.setdefault("numpy._core.numeric", np_numeric)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit Phase 1 v2 GEE logistic model.")
    parser.add_argument("--input-path", type=Path, default=INPUT_PATH)
    parser.add_argument("--output-prefix", default="phase_1_v2_gee")
    return parser.parse_args()


def load_data(input_path: Path) -> pd.DataFrame:
    install_numpy_pickle_compat()
    df = pd.read_pickle(input_path).copy()
    df["Year_centered"] = df["Year"] - df["Year"].min()

    # v2 engineered features
    df["FBS_hinge_100"] = (df["FBS"] - 100).clip(lower=0)   # NaN propagates from FBS
    df["FBS_hinge_125"] = (df["FBS"] - 125).clip(lower=0)
    df["Year_centered_sq"] = df["Year_centered"] ** 2
    df["FBS_x_Age"] = df["FBS"] * df["Age"]                  # NaN propagates from FBS

    if "Target_AtRisk_Status" not in df.columns:
        df["Target_AtRisk_Status"] = df["Next_Year_AtRisk_Status"]
    return df


def split_by_patient(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return apply_canonical_split(df)


def get_continuous_features(df: pd.DataFrame) -> list[str]:
    base_features = [f for f in BASE_CONTINUOUS_FEATURES if f in df.columns]
    history_features = [
        col for col in df.columns
        if "_hist_" in col
        and col.endswith(HISTORY_FEATURE_PATTERNS)
        and pd.api.types.is_numeric_dtype(df[col])
    ]
    return base_features + sorted(history_features)


def prepare_features(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train = train_df.copy()
    test = test_df.copy()
    z_features: list[str] = []

    for feature in get_continuous_features(train):
        median = train[feature].median()
        if pd.isna(median):
            median = 0.0
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
    terms = (
        z_features
        + [f"{f}_missing" for f in MISSING_INDICATOR_FEATURES]
        + ["is_missing_last_year_filled"]
        + [f"C({f})" for f in CATEGORICAL_FEATURES]
    )
    return "target ~ " + " + ".join(terms)


def classification_metrics(
    y_true: np.ndarray, probability: np.ndarray, threshold: float
) -> dict[str, float]:
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


def coefficients_table(result: object) -> pd.DataFrame:
    table = pd.DataFrame({
        "feature": result.params.index,
        "coefficient": result.params.to_numpy(),
        "robust_std_error": result.bse.to_numpy(),
        "p_value": result.pvalues.to_numpy(),
    })
    table["odds_ratio"] = np.exp(table["coefficient"])
    table["or_ci_low"] = np.exp(table["coefficient"] - 1.96 * table["robust_std_error"])
    table["or_ci_high"] = np.exp(table["coefficient"] + 1.96 * table["robust_std_error"])
    table["abs_log_or"] = table["coefficient"].abs()
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
    dependence_parameter: float,
) -> str:
    lines = [
        "# Phase 1 v2 GEE Logistic Risk Model",
        "",
        "## Scope",
        "Improved GEE model using `statsmodels` with Binomial family and exchangeable "
        "within-patient working correlation. Feature improvements from Phase 0.2 EDA: "
        "removed `pulse_pressure` (VIF=225), added FBS hockey-stick features at clinical "
        "thresholds (100 mg/dL pre-DM, 125 mg/dL DM), quadratic time trend, and "
        "FBS×Age interaction.",
        "",
        "## Data Split",
        f"Grouped by `PatientId`: `{train_df['PatientId'].nunique():,}` train patients, "
        f"`{test_df['PatientId'].nunique():,}` test patients (seed={RANDOM_SEED}).",
        "",
        "## Formula",
        f"`{formula}`",
        "",
        f"Estimated exchangeable dependence parameter: `{dependence_parameter:.4f}`.",
        "",
        "## Metrics",
        markdown_table(metrics_df),
        "",
        "## Most Influential Odds Ratios",
        "Continuous features are median-imputed and standardized on the training set.",
        markdown_table(coef_df[coef_df["feature"] != "Intercept"], max_rows=30),
        "",
        "## Test Calibration Deciles",
        markdown_table(cal_df),
        "",
        "## v2 Feature Changes vs v1",
        "| Change | Feature | Reason |",
        "| --- | --- | --- |",
        "| Removed | `pulse_pressure` | VIF=225; equals BL_pres1 − BL_pres2 by construction |",
        "| Added | `FBS_hinge_100` | Hockey-stick at pre-DM threshold (100 mg/dL) |",
        "| Added | `FBS_hinge_125` | Hockey-stick at DM threshold (125 mg/dL) |",
        "| Added | `Year_centered_sq` | Non-linear time trend (Ljung-Box p=0.03) |",
        "| Added | `FBS_x_Age` | FBS × Age interaction (top-2 predictors) |",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data(args.input_path)
    raw_train, raw_test = split_by_patient(df)
    train_df, test_df, z_features = prepare_features(raw_train, raw_test)
    formula = build_formula(z_features)

    model = smf.gee(
        formula=formula,
        groups="PatientId",
        data=train_df,
        family=sm.families.Binomial(),
        cov_struct=sm.cov_struct.Exchangeable(),
    )
    result = model.fit(maxiter=100)

    train_probability = result.predict(train_df)
    test_probability = result.predict(test_df)
    y_train = train_df["target"].to_numpy()
    y_test = test_df["target"].to_numpy()
    threshold = float(y_train.mean())

    metrics_df = pd.DataFrame([
        {"split": "train", "calibration_method": "raw",
         "threshold_strategy": "train_positive_rate",
         **classification_metrics(y_train, train_probability, threshold)},
        {"split": "test", "calibration_method": "raw",
         "threshold_strategy": "train_positive_rate",
         **classification_metrics(y_test, test_probability, threshold)},
    ])
    coef_df = coefficients_table(result)
    cal_df = calibration_table(y_test, test_probability)

    predictions = test_df[["PatientId", "Year", "target_year", "target"]].copy()
    predictions = predictions.rename(columns={"target": "Target_AtRisk_Status"})
    predictions["predicted_probability"] = test_probability

    prefix = args.output_prefix
    coef_df.to_csv(OUT_DIR / f"{prefix}_coefficients.csv", index=False)
    metrics_df.to_csv(OUT_DIR / f"{prefix}_metrics.csv", index=False)
    predictions.to_csv(OUT_DIR / f"{prefix}_test_predictions.csv", index=False)
    cal_df.to_csv(OUT_DIR / f"{prefix}_test_calibration.csv", index=False)

    dependence_parameter = float(np.asarray(result.cov_struct.dep_params).ravel()[0])
    report = write_report(train_df, test_df, formula, metrics_df, coef_df, cal_df, dependence_parameter)
    (OUT_DIR / f"{prefix}_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
