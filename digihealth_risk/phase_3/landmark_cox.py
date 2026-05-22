"""Phase 3.2 v2 landmark survival — improved feature engineering.

Changes from v1 (phase_3_2_landmark_survival.py) driven by Phase 0.2 EDA:
  - Removed `pulse_pressure` (VIF=225; = BL_pres1 − BL_pres2 by construction)
  - Added `Year_centered_sq` (U-shaped temporal risk trend; Ljung-Box p=0.03)
  - Added `FBS_x_Age` interaction (Phase 0.2 top cross-lag predictor)
  - Added `MAX_FBS_x_Age` interaction (MAX_FBS_up_to_year × Age; cross-lag r=0.582)
  - NOTE: FBS_hinge_100 / FBS_hinge_125 excluded despite plan — the modeling table
    only contains non-at-risk rows (FBS ≤ 100 mg/dL by construction), making both
    features all-zero (std=0) and causing a singular Cox Hessian.

Run from the repository root:
    python digihealth_risk/phase_3/landmark_cox.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.duration.hazard_regression import PHReg


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from digihealth_risk.phase_3.survival_utils import concordance_index, hazard_ratio_table, markdown_table
from digihealth_risk.utils.patient_split import apply_canonical_split


LONG_PATH = ROOT / "digihealth_risk" / "phase_0" / "outputs" / "patient_year_long.pkl"
OUT_DIR = ROOT / "digihealth_risk" / "phase_3" / "outputs"
RANDOM_SEED = 20260501
BINARY_HORIZONS = [1, 2, 3, 4, 5]
END_YEAR = 2016

NUMERIC_FEATURES = [
    "Year",
    "Year_centered_sq",        # v2: U-shaped temporal trend
    "Age",
    "FBS",
    # FBS_hinge_100 / FBS_hinge_125: all-zero in modeling table (non-at-risk rows have
    #   FBS ≤ 100 mg/dL by construction) → std=0 → singular Cox Hessian — excluded
    "FBS_x_Age",               # v2: FBS × Age interaction
    "BMI",
    "Pulse",
    "BL_pres1",
    "BL_pres2",
    "Waist",
    # pulse_pressure removed (VIF=225; = BL_pres1 − BL_pres2 by construction)
    "MAX_FBS_up_to_year",
    "MAX_FBS_x_Age",           # v2: MAX_FBS_up_to_year × Age (cross-lag r=0.582)
    "clinical_observed_count",
    "has_fbs_this_year",
    "years_since_last_fbs",
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


def load_long_table() -> pd.DataFrame:
    return pd.read_pickle(LONG_PATH).copy()


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Year_centered" not in df.columns:
        df["Year_centered"] = df["Year"] - df["Year"].min()
    if "Year_centered_sq" not in df.columns:
        df["Year_centered_sq"] = df["Year_centered"] ** 2
    if "FBS_hinge_100" not in df.columns:
        df["FBS_hinge_100"] = (df["FBS"] - 100).clip(lower=0)
    if "FBS_hinge_125" not in df.columns:
        df["FBS_hinge_125"] = (df["FBS"] - 125).clip(lower=0)
    if "FBS_x_Age" not in df.columns:
        df["FBS_x_Age"] = df["FBS"] * df["Age"]
    if "MAX_FBS_x_Age" not in df.columns:
        df["MAX_FBS_x_Age"] = df["MAX_FBS_up_to_year"] * df["Age"]
    return df


def split_by_patient(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return apply_canonical_split(df)


def first_event_and_censor(long_df: pd.DataFrame) -> pd.DataFrame:
    patient_groups = long_df.groupby("PatientId", sort=False)
    first_event = patient_groups.apply(
        lambda group: group.loc[group["AtRisk_current_year"].eq(1), "Year"].min(),
        include_groups=False,
    ).rename("first_atrisk_year_landmark")
    last_observed = patient_groups.apply(
        lambda group: group.loc[group["AtRisk_current_year"].notna(), "Year"].max(),
        include_groups=False,
    ).rename("last_observed_year_landmark")
    return pd.concat([first_event, last_observed], axis=1).reset_index()


def build_landmark_table(long_df: pd.DataFrame) -> pd.DataFrame:
    outcomes = first_event_and_censor(long_df)
    table = long_df.merge(outcomes, on="PatientId", how="left")
    table = table[table["AtRisk_current_year"].eq(0)].copy()
    table = table[table["Year"].lt(END_YEAR)].copy()

    table["event_after_landmark"] = (
        table["first_atrisk_year_landmark"].notna()
        & table["first_atrisk_year_landmark"].gt(table["Year"])
    )
    table["event"] = table["event_after_landmark"].astype(int)
    table["event_or_censor_year"] = np.where(
        table["event"].eq(1),
        table["first_atrisk_year_landmark"],
        table["last_observed_year_landmark"],
    )
    table["duration"] = (table["event_or_censor_year"] - table["Year"]).clip(lower=1).astype(float)
    table = table[table["event_or_censor_year"].gt(table["Year"])].copy()

    status_lookup = long_df.set_index(["PatientId", "Year"])["AtRisk_current_year"]
    for horizon in BINARY_HORIZONS:
        target_index = pd.MultiIndex.from_arrays(
            [table["PatientId"], table["Year"] + horizon],
            names=["PatientId", "Year"],
        )
        table[f"Target_AtRisk_Status_horizon_{horizon}"] = status_lookup.reindex(target_index).to_numpy()
    return table


def make_preprocessor() -> ColumnTransformer:
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first")
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False, drop="first")

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", encoder),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )


def fit_cox(train_df: pd.DataFrame) -> tuple[object, ColumnTransformer, list[str]]:
    preprocessor = make_preprocessor()
    x_train = preprocessor.fit_transform(train_df)
    feature_names = [str(name) for name in preprocessor.get_feature_names_out()]
    model = PHReg(
        endog=train_df["duration"].to_numpy(),
        exog=x_train,
        status=train_df["event"].to_numpy(),
        ties="efron",
    )
    result = model.fit(groups=train_df["PatientId"])
    return result, preprocessor, feature_names


def risk_score(result: object, preprocessor: ColumnTransformer, df: pd.DataFrame) -> np.ndarray:
    x = preprocessor.transform(df)
    return np.asarray(x @ result.params, dtype=float)


def baseline_survival_at_horizons(train_df: pd.DataFrame, train_score: np.ndarray) -> dict[int, float]:
    exp_score = np.exp(train_score)
    baseline_hazard = 0.0
    survival_by_time: dict[int, float] = {}

    event_times = sorted(train_df.loc[train_df["event"].eq(1), "duration"].unique())
    for event_time in event_times:
        events_at_time = int(((train_df["duration"].eq(event_time)) & train_df["event"].eq(1)).sum())
        risk_set = train_df["duration"].ge(event_time).to_numpy()
        risk_sum = float(exp_score[risk_set].sum())
        if risk_sum > 0:
            baseline_hazard += events_at_time / risk_sum
        survival_by_time[int(event_time)] = float(np.exp(-baseline_hazard))

    result = {}
    for horizon in BINARY_HORIZONS:
        eligible = [time for time in survival_by_time if time <= horizon]
        result[horizon] = survival_by_time[max(eligible)] if eligible else 1.0
    return result


def event_probability_by_horizon(score: np.ndarray, baseline_survival: dict[int, float]) -> dict[int, np.ndarray]:
    exp_score = np.exp(score)
    return {horizon: 1.0 - np.power(s0, exp_score) for horizon, s0 in baseline_survival.items()}


def target_column_for_horizon(horizon: int) -> str:
    return f"Target_AtRisk_Status_horizon_{horizon}"


def binary_label_for_horizon(df: pd.DataFrame, horizon: int) -> np.ndarray:
    return df[target_column_for_horizon(horizon)].astype(int).to_numpy()


def binary_metrics(y_true: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float]:
    prediction = probability >= threshold
    return {
        "rows": float(len(y_true)),
        "positives": float(y_true.sum()),
        "positive_rate": float(y_true.mean()),
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "pr_auc": float(average_precision_score(y_true, probability)),
        "brier": float(brier_score_loss(y_true, probability)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "specificity": float(((~prediction) & (y_true == 0)).sum() / max((y_true == 0).sum(), 1)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
    }


def binary_horizon_evaluation(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_probabilities: dict[int, np.ndarray],
    test_probabilities: dict[int, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    prediction_frames = []
    base_predictions = test_df[
        [
            "PatientId",
            "Year",
            "duration",
            "event",
            "first_atrisk_year_landmark",
            "last_observed_year_landmark",
        ]
    ].copy()

    for horizon in BINARY_HORIZONS:
        target_column = target_column_for_horizon(horizon)
        train_mask = train_df[target_column].notna().to_numpy()
        test_mask = test_df[target_column].notna().to_numpy()

        y_train = binary_label_for_horizon(train_df.loc[train_mask], horizon)
        y_test = binary_label_for_horizon(test_df.loc[test_mask], horizon)
        threshold = float(y_train.mean())

        for split, y_true, probability in [
            ("train", y_train, train_probabilities[horizon][train_mask]),
            ("test", y_test, test_probabilities[horizon][test_mask]),
        ]:
            metric_rows.append(
                {
                    "split": split,
                    "horizon_years": horizon,
                    "calibration_method": "raw",
                    "threshold_strategy": "train_positive_rate",
                    **binary_metrics(y_true, probability, threshold),
                }
            )

        pred = base_predictions.loc[test_mask].copy()
        pred["horizon_years"] = horizon
        pred["target_year"] = pred["Year"] + horizon
        pred["Target_AtRisk_Status"] = y_test
        pred["predicted_probability"] = test_probabilities[horizon][test_mask]
        prediction_frames.append(pred)

    return pd.DataFrame(metric_rows), pd.concat(prediction_frames, ignore_index=True)


def write_report(
    landmark_df: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cox_metrics: pd.DataFrame,
    binary_metrics_df: pd.DataFrame,
    hr_df: pd.DataFrame,
) -> str:
    lines = [
        "# Phase 3.2 v2 Landmark Survival Report",
        "",
        "## Scope",
        "Improved rolling landmark Cox model. Feature changes from Phase 0.2 EDA: "
        "`pulse_pressure` removed (VIF=225), `Year_centered_sq` captures the U-shaped temporal trend, "
        "and interaction terms were added. FBS hockey-stick thresholds were evaluated but excluded from the final Cox fit.",
        "",
        "## v2 Feature Changes vs v1",
        "| Change | Feature | Reason |",
        "| --- | --- | --- |",
        "| Removed | `pulse_pressure` | VIF=225; equals BL_pres1 − BL_pres2 by construction |",
        "| Excluded | `FBS_hinge_100` | All-zero in non-at-risk landmark rows; singular Cox Hessian |",
        "| Excluded | `FBS_hinge_125` | All-zero in non-at-risk landmark rows; singular Cox Hessian |",
        "| Added | `Year_centered_sq` | U-shaped temporal risk trend (Ljung-Box p=0.03) |",
        "| Added | `FBS_x_Age` | Phase 0.2 top cross-lag interaction |",
        "| Added | `MAX_FBS_x_Age` | MAX_FBS_up_to_year × Age; cross-lag r=0.582 |",
        "",
        "## Data Summary",
        f"Landmark rows: `{len(landmark_df):,}`.",
        f"Train rows: `{len(train_df):,}` across `{train_df['PatientId'].nunique():,}` patients.",
        f"Test rows: `{len(test_df):,}` across `{test_df['PatientId'].nunique():,}` patients.",
        "",
        "## Cox Ranking Metrics",
        markdown_table(cox_metrics),
        "",
        "## Fixed-Horizon Binary Metrics",
        markdown_table(binary_metrics_df[binary_metrics_df["split"].eq("test")]),
        "",
        "## Top Hazard Ratios",
        markdown_table(hr_df, max_rows=25),
        "",
        "## Notes",
        "- Cox fit uses cluster-robust groups by `PatientId`.",
        "- `Year` is included as a covariate (within-year HRs).",
        "- Fixed-horizon thresholds use the observed-target training positive rate for each horizon.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    long_df = load_long_table()
    landmark_df = build_landmark_table(long_df)
    landmark_df = engineer_features(landmark_df)  # add FBS hinges, Year_centered_sq, interactions
    train_df, test_df = split_by_patient(landmark_df)

    result, preprocessor, feature_names = fit_cox(train_df)
    train_score = risk_score(result, preprocessor, train_df)
    test_score = risk_score(result, preprocessor, test_df)
    baseline_survival = baseline_survival_at_horizons(train_df, train_score)
    train_probabilities = event_probability_by_horizon(train_score, baseline_survival)
    test_probabilities = event_probability_by_horizon(test_score, baseline_survival)

    cox_metrics = pd.DataFrame(
        [
            {
                "split": "train",
                "rows": len(train_df),
                "patients": train_df["PatientId"].nunique(),
                "events": int(train_df["event"].sum()),
                "event_rate": float(train_df["event"].mean()),
                "c_index": concordance_index(
                    train_df["duration"].to_numpy(),
                    train_df["event"].to_numpy(),
                    train_score,
                ),
            },
            {
                "split": "test",
                "rows": len(test_df),
                "patients": test_df["PatientId"].nunique(),
                "events": int(test_df["event"].sum()),
                "event_rate": float(test_df["event"].mean()),
                "c_index": concordance_index(
                    test_df["duration"].to_numpy(),
                    test_df["event"].to_numpy(),
                    test_score,
                ),
            },
        ]
    )
    binary_metrics_df, predictions = binary_horizon_evaluation(
        train_df,
        test_df,
        train_probabilities,
        test_probabilities,
    )
    hr_df = hazard_ratio_table(result, feature_names)

    landmark_df.to_pickle(OUT_DIR / "phase_3_2_v2_landmark_table.pkl")
    landmark_df.head(1000).to_csv(OUT_DIR / "phase_3_2_v2_landmark_table_sample.csv", index=False)
    cox_metrics.to_csv(OUT_DIR / "phase_3_2_v2_cox_metrics.csv", index=False)
    binary_metrics_df.to_csv(OUT_DIR / "phase_3_2_v2_binary_horizon_metrics.csv", index=False)
    predictions.to_csv(OUT_DIR / "phase_3_2_v2_test_predictions.csv", index=False)
    hr_df.to_csv(OUT_DIR / "phase_3_2_v2_hazard_ratios.csv", index=False)

    report = write_report(landmark_df, train_df, test_df, cox_metrics, binary_metrics_df, hr_df)
    (OUT_DIR / "phase_3_2_v2_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
