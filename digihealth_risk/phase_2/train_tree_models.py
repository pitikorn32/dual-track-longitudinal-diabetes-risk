"""Phase 2 v2 tree-model experiments — improved feature engineering.

Changes from v1 (phase_2_train_tree_models.py) driven by Phase 0.2 EDA:
  - Removed `pulse_pressure` (VIF=225; = BL_pres1 − BL_pres2 by construction)
  - Added FBS hockey-stick features at clinical thresholds (100 mg/dL pre-DM,
    125 mg/dL DM)
  - Added `Year_centered_sq` for the non-linear temporal trend confirmed in
    Phase 1 v2 (U-shaped; large coefficient in both GEE and logistic v2)
  - Added `FBS_x_Age` and `MAX_FBS_x_Age` interactions (Phase 0.2 cross-lag)

All five model types, hyperparameters, and output schema are unchanged.

Examples:
    python digihealth_risk/phase_2/train_tree_models.py
    python digihealth_risk/phase_2/train_tree_models.py \\
      --input-path digihealth_risk/phase_0/outputs/phase_0_modeling_table_horizon_3_history_5.pkl
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from digihealth_risk.utils.patient_split import apply_canonical_split  # noqa: E402

PHASE0_OUT = ROOT / "digihealth_risk" / "phase_0" / "outputs"
PHASE1_OUT = ROOT / "digihealth_risk" / "phase_1" / "outputs"
OUT_DIR = ROOT / "digihealth_risk" / "phase_2" / "outputs"
RANDOM_SEED = 20260501
PERMUTATION_SAMPLE_SIZE = 3000

DEFAULT_INPUT_PATHS = [
    PHASE0_OUT / "phase_0_modeling_table.pkl",
    PHASE0_OUT / "phase_0_modeling_table_horizon_3_history_5.pkl",
]

LEAKAGE_OR_METADATA_COLUMNS = {
    "PatientId",
    "date_of_birth",
    "DM_status_up_to_year",
    "AtRisk_current_year",
    "first_atrisk_year",
    "target_year",
    "Target_AtRisk_Status",
    "Next_Year_AtRisk_Status",
    "prediction_horizon_years",
    "history_window_years",
    "history_start_year",
    "pulse_pressure",          # v2: removed — VIF=225, = BL_pres1 - BL_pres2 by construction
}


def install_numpy_pickle_compat() -> None:
    import sys
    import numpy.core as np_core
    import numpy.core.multiarray as np_multiarray
    import numpy.core.numeric as np_numeric

    sys.modules.setdefault("numpy._core", np_core)
    sys.modules.setdefault("numpy._core.multiarray", np_multiarray)
    sys.modules.setdefault("numpy._core.numeric", np_numeric)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Phase 2 v2 tree models.")
    parser.add_argument("--input-path", type=Path, action="append",
                        help="Phase 0 modeling table. Can be passed multiple times.")
    parser.add_argument("--models", nargs="+",
                        default=["histgb", "random_forest", "xgboost", "lightgbm", "catboost"],
                        choices=["histgb", "random_forest", "xgboost", "lightgbm", "catboost"])
    parser.add_argument("--skip-phase1-comparison", action="store_true")
    parser.add_argument("--use-class-weights", action="store_true")
    parser.add_argument("--output-prefix", default="phase_2_v2")
    parser.add_argument("--include-phase2-baseline", action="store_true")
    return parser.parse_args()


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add v2 engineered features. Safe to call repeatedly — skips existing columns."""
    df = df.copy()
    if "Year_centered" not in df.columns:
        df["Year_centered"] = df["Year"] - df["Year"].min()
    if "Year_centered_sq" not in df.columns:
        df["Year_centered_sq"] = df["Year_centered"] ** 2
    if "FBS_hinge_100" not in df.columns:
        df["FBS_hinge_100"] = (df["FBS"] - 100).clip(lower=0)   # NaN propagates from FBS
    if "FBS_hinge_125" not in df.columns:
        df["FBS_hinge_125"] = (df["FBS"] - 125).clip(lower=0)
    if "FBS_x_Age" not in df.columns:
        df["FBS_x_Age"] = df["FBS"] * df["Age"]
    if "MAX_FBS_x_Age" not in df.columns:
        df["MAX_FBS_x_Age"] = df["MAX_FBS_up_to_year"] * df["Age"]
    return df


def slugify_path(path: Path) -> str:
    name = path.stem
    name = name.replace("phase_0_modeling_table", "phase2")
    name = re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_")
    return name or "dataset"


def load_table(path: Path) -> pd.DataFrame:
    install_numpy_pickle_compat()
    df = pd.read_pickle(path).copy()
    if "Target_AtRisk_Status" not in df.columns:
        df["Target_AtRisk_Status"] = df["Next_Year_AtRisk_Status"]
    return df


def split_by_patient(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return apply_canonical_split(df)


def get_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    candidates = [
        col
        for col in df.columns
        if col not in LEAKAGE_OR_METADATA_COLUMNS and not df[col].isna().all()
    ]
    numeric_features = [
        col for col in candidates
        if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col])
    ]
    categorical_features = [
        col for col in candidates
        if col not in numeric_features and not pd.api.types.is_datetime64_any_dtype(df[col])
    ]
    return numeric_features, categorical_features


def make_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    return ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_features),
            (
                "cat",
                Pipeline(steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", encoder),
                ]),
                categorical_features,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_model(model_name: str, scale_pos_weight: float, use_class_weights: bool) -> Any:
    effective_scale_pos_weight = scale_pos_weight if use_class_weights else 1.0

    if model_name == "histgb":
        return HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=250, max_leaf_nodes=31, l2_regularization=0.01,
            early_stopping=True, class_weight="balanced" if use_class_weights else None,
            random_state=RANDOM_SEED,
        )
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=400, min_samples_leaf=20, max_features="sqrt",
            class_weight="balanced_subsample" if use_class_weights else None,
            n_jobs=-1, random_state=RANDOM_SEED,
        )
    if model_name == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=400, max_depth=3, learning_rate=0.03, subsample=0.9,
            colsample_bytree=0.9, reg_lambda=2.0, objective="binary:logistic",
            eval_metric="logloss", scale_pos_weight=effective_scale_pos_weight,
            n_jobs=-1, random_state=RANDOM_SEED,
        )
    if model_name == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=40,
            subsample=0.9, colsample_bytree=0.9,
            class_weight="balanced" if use_class_weights else None,
            random_state=RANDOM_SEED, n_jobs=-1, verbose=-1,
        )
    if model_name == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(
            iterations=400, depth=4, learning_rate=0.03, loss_function="Logloss",
            class_weights=[1.0, scale_pos_weight] if use_class_weights else None,
            random_seed=RANDOM_SEED, allow_writing_files=False, verbose=False,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def classification_metrics(y_true: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float]:
    prediction = probability >= threshold
    return {
        "calibration_method": "raw",
        "threshold_strategy": "train_positive_rate",
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


def transformed_feature_names(pipeline: Pipeline) -> list[str]:
    return [str(n) for n in pipeline.named_steps["preprocessor"].get_feature_names_out()]


def feature_importance(
    pipeline: Pipeline, model_name: str, x_test: pd.DataFrame, y_test: pd.Series
) -> pd.DataFrame:
    model = pipeline.named_steps["model"]

    if hasattr(model, "feature_importances_"):
        names = transformed_feature_names(pipeline)
        importance = np.asarray(model.feature_importances_, dtype=float)
        return pd.DataFrame({"feature": names, "importance": importance}).sort_values(
            "importance", ascending=False
        )

    if model_name == "histgb":
        sample_size = min(PERMUTATION_SAMPLE_SIZE, len(x_test))
        x_sample = x_test.sample(n=sample_size, random_state=RANDOM_SEED)
        y_sample = y_test.loc[x_sample.index]
        result = permutation_importance(
            pipeline, x_sample, y_sample,
            scoring="average_precision", n_repeats=2,
            random_state=RANDOM_SEED, n_jobs=-1,
        )
        return pd.DataFrame({
            "feature": x_test.columns,
            "importance": result.importances_mean,
            "importance_std": result.importances_std,
        }).sort_values("importance", ascending=False)

    return pd.DataFrame(columns=["feature", "importance"])


def train_one_model(
    model_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    use_class_weights: bool,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    feature_columns = numeric_features + categorical_features
    x_train = train_df[feature_columns].copy()
    x_test = test_df[feature_columns].copy()
    y_train = train_df["Target_AtRisk_Status"].astype(int).to_numpy()
    y_test_series = test_df["Target_AtRisk_Status"].astype(int)
    y_test = y_test_series.to_numpy()

    positives = y_train.sum()
    negatives = len(y_train) - positives
    scale_pos_weight = float(negatives / positives) if positives else 1.0

    pipeline = Pipeline(steps=[
        ("preprocessor", make_preprocessor(numeric_features, categorical_features)),
        ("model", build_model(model_name, scale_pos_weight, use_class_weights)),
    ])
    pipeline.fit(x_train, y_train)

    train_probability = pipeline.predict_proba(x_train)[:, 1]
    test_probability  = pipeline.predict_proba(x_test)[:, 1]
    threshold = float(y_train.mean())

    test_metrics  = {"model": model_name, "split": "test",
                     **classification_metrics(y_test, test_probability, threshold)}
    train_metrics = {"model": model_name, "split": "train",
                     **classification_metrics(y_train, train_probability, threshold)}

    predictions = test_df[["PatientId", "Year", "target_year", "Target_AtRisk_Status"]].copy()
    predictions["model"] = model_name
    predictions["predicted_probability"] = test_probability

    importance = feature_importance(pipeline, model_name, x_test, y_test_series)
    importance.insert(0, "model", model_name)
    return train_metrics | {"_test_metrics": test_metrics}, predictions, importance


def read_phase1_metrics() -> pd.DataFrame:
    rows = []
    files = {
        "phase1_gee_v2_default": PHASE1_OUT / "phase_1_v2_gee_metrics.csv",
        "phase1_gee_default":    PHASE1_OUT / "phase_1_gee_metrics.csv",
    }
    for model_name, path in files.items():
        if not path.exists():
            continue
        df = pd.read_csv(path)
        test = df[df["split"] == "test"].copy()
        test["model"] = model_name
        rows.append(test)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def read_phase2_baseline_metrics(output_prefix: str) -> pd.DataFrame:
    if output_prefix == "phase_2":
        return pd.DataFrame()
    path = OUT_DIR / "phase_2_metrics.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    return df[df["split"] == "test"].copy()


def markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if max_rows is not None:
        df = df.head(max_rows)
    display = df.copy()
    for col in display.select_dtypes(include=[np.number]).columns:
        display[col] = display[col].map(lambda x: "" if pd.isna(x) else f"{x:.4g}")
    cols = display.columns.tolist()
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def write_report(
    metrics_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    inputs: list[Path],
    use_class_weights: bool,
) -> str:
    lines = [
        "# Phase 2 v2 Tree Model Report",
        "",
        "## Scope",
        "Improved Phase 2 tree classifiers on Phase 0 modeling tables. "
        "Feature improvements from Phase 0.2 EDA: removed `pulse_pressure` (VIF=225), "
        "added FBS hockey-stick features (100 mg/dL pre-DM, 125 mg/dL DM), "
        "quadratic time trend `Year_centered_sq`, and `FBS_x_Age` / `MAX_FBS_x_Age` interactions.",
        f"Class weighting enabled: `{use_class_weights}`.",
        "",
        "## v2 Feature Changes",
        "| Change | Feature | Reason |",
        "| --- | --- | --- |",
        "| Removed | `pulse_pressure` | VIF=225; = BL_pres1 − BL_pres2 by construction |",
        "| Added | `FBS_hinge_100` | Hockey-stick at pre-DM threshold (100 mg/dL) |",
        "| Added | `FBS_hinge_125` | Hockey-stick at DM threshold (125 mg/dL) |",
        "| Added | `Year_centered_sq` | Non-linear time trend (U-shaped, confirmed Phase 1 v2) |",
        "| Added | `FBS_x_Age` | FBS × Age interaction (Phase 0.2 top-2 predictors) |",
        "| Added | `MAX_FBS_x_Age` | MAX_FBS × Age interaction (cross-lag r=0.582) |",
        "",
        "## Inputs",
    ]
    for path in inputs:
        resolved = path.resolve()
        try:
            label = resolved.relative_to(ROOT)
        except ValueError:
            label = resolved
        lines.append(f"- `{label}`")

    test_cols = [
        "dataset", "model", "rows", "positives", "positive_rate",
        "roc_auc", "pr_auc", "brier", "precision", "recall", "specificity", "f1",
    ]
    ranked = metrics_df[metrics_df["split"] == "test"].sort_values(
        ["dataset", "pr_auc", "roc_auc"], ascending=[True, False, False]
    )
    lines.extend(["", "## Phase 2 v2 Test Metrics",
                  markdown_table(ranked[[c for c in test_cols if c in ranked.columns]])])

    if not comparison_df.empty:
        compare_cols = [c for c in test_cols if c in comparison_df.columns]
        lines.extend(["", "## Phase 1 + Phase 2 Comparison",
                      markdown_table(comparison_df[compare_cols])])

    lines.extend([
        "",
        "## Notes",
        "- Splits are grouped by `PatientId` with seed `20260501`.",
        "- Threshold-based metrics use the training positive rate as the probability cutoff.",
        "- `calibration_method=raw`, `threshold_strategy=train_positive_rate` for all rows.",
        "- Class weighting is off by default to preserve probability calibration.",
        "",
    ])
    return "\n".join(lines)


def run_dataset(
    path: Path,
    models: list[str],
    use_class_weights: bool,
) -> tuple[pd.DataFrame, list[pd.DataFrame], list[pd.DataFrame]]:
    dataset = slugify_path(path)
    df = load_table(path)
    df = engineer_features(df)          # v2: add engineered features before splitting
    train_df, test_df = split_by_patient(df)
    numeric_features, categorical_features = get_feature_columns(train_df)

    print(
        f"Dataset={dataset} rows={len(df)} features={len(numeric_features)+len(categorical_features)} "
        f"numeric={len(numeric_features)} categorical={len(categorical_features)}"
    )

    metrics_rows: list[dict] = []
    predictions: list[pd.DataFrame] = []
    importances: list[pd.DataFrame] = []

    for model_name in models:
        print(f"Training {dataset}/{model_name}")
        metrics_bundle, pred_df, importance_df = train_one_model(
            model_name, train_df, test_df, numeric_features, categorical_features, use_class_weights
        )
        test_metrics = metrics_bundle.pop("_test_metrics")
        for metrics in [metrics_bundle, test_metrics]:
            metrics["dataset"] = dataset
            metrics["horizon_years"] = int(df["prediction_horizon_years"].iloc[0])
            metrics["history_years"] = int(df["history_window_years"].iloc[0])
            metrics_rows.append(metrics)

        pred_df.insert(0, "dataset", dataset)
        importance_df.insert(0, "dataset", dataset)
        predictions.append(pred_df)
        importances.append(importance_df)

    return pd.DataFrame(metrics_rows), predictions, importances


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    input_paths = args.input_path or [p for p in DEFAULT_INPUT_PATHS if p.exists()]
    if not input_paths:
        raise FileNotFoundError("No Phase 0 modeling tables found.")

    all_metrics, all_predictions, all_importances = [], [], []
    for path in input_paths:
        metrics_df, predictions, importances = run_dataset(path, args.models, args.use_class_weights)
        all_metrics.append(metrics_df)
        all_predictions.extend(predictions)
        all_importances.extend(importances)

    metrics     = pd.concat(all_metrics, ignore_index=True)
    predictions = pd.concat(all_predictions, ignore_index=True)
    importances = pd.concat(all_importances, ignore_index=True)

    metrics.to_csv(OUT_DIR / f"{args.output_prefix}_metrics.csv", index=False)
    predictions.to_csv(OUT_DIR / f"{args.output_prefix}_test_predictions.csv", index=False)
    importances.to_csv(OUT_DIR / f"{args.output_prefix}_feature_importance.csv", index=False)

    comparison_parts = []
    phase1 = pd.DataFrame() if args.skip_phase1_comparison else read_phase1_metrics()
    if not phase1.empty:
        phase1["dataset"] = np.where(
            phase1["model"].str.contains("horizon_3_history_5"),
            "phase2_horizon_3_history_5", "phase2",
        )
        phase1["horizon_years"] = np.where(phase1["dataset"].str.contains("horizon_3"), 3, 1)
        phase1["history_years"] = np.where(phase1["dataset"].str.contains("history_5"), 5, 1)
        comparison_parts.append(phase1)

    if args.include_phase2_baseline:
        phase2_baseline = read_phase2_baseline_metrics(args.output_prefix)
        if not phase2_baseline.empty:
            comparison_parts.append(phase2_baseline)

    comparison_parts.append(metrics[metrics["split"] == "test"])
    comparison = pd.concat(comparison_parts, ignore_index=True)
    comparison = comparison.sort_values(
        ["dataset", "pr_auc", "roc_auc"], ascending=[True, False, False]
    )
    comparison.to_csv(OUT_DIR / f"{args.output_prefix}_comparison_metrics.csv", index=False)

    report = write_report(metrics, comparison, input_paths, args.use_class_weights)
    (OUT_DIR / f"{args.output_prefix}_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
