"""Phase 8 — Logistic-only alternative screening model build.

Trains and exports 15 screening-track logistic regression artifacts (one per
horizon N x history M cell of the canonical 5x3 grid). Supports a `--no-year`
flag that produces a calendar-time-invariant variant via the phase_7
year-features ablation.

Two consecutive invocations (with and without `--no-year`) produce the full
30-artifact alternative deployment set:

    python digihealth_risk/phase_8/export_logistic_models.py
    python digihealth_risk/phase_8/export_logistic_models.py --no-year

Outputs (under `digihealth_risk/phase_8/outputs/`):

    models/screening_logistic_n{N}_m{M}.joblib          (with-Year, 15 files)
    model_registry.json                                  (with-Year)
    deployment_metrics.csv                               (with-Year)

    models_no_year/screening_logistic_n{N}_m{M}.joblib   (no-Year, 15 files)
    model_registry_no_year.json                          (no-Year)
    deployment_metrics_no_year.csv                       (no-Year)

Why this exists:

The phase 6 screening track uses a mixed-family per-horizon winner mix
(CatBoost at N=1, XGBoost at N=3, Logistic at N=2/4/5). Phase 8 is the
dependency-light alternative: a single family (logistic) at every horizon, no
catboost/xgboost/interpret runtime requirement, and a uniform serialization
profile. The artifact schema matches phase 6's logistic artifact exactly so
wiring this into the FastAPI later is purely a routing change.

Run from the repository root.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Import phase_6 helpers and phase_2 utilities. The no-Year monkey patch is
# applied inside main() before any feature engineering happens; it mutates the
# already-imported phase_2 module in place, so the imports below are safe.
from digihealth_risk.phase_2.train_tree_models import (  # noqa: E402
    classification_metrics,
    engineer_features,
    load_table,
)
from digihealth_risk.phase_6.export_models import (  # noqa: E402
    compute_intervention_presets,
    fit_logistic_artifact,
    model_key,
    phase0_path,
    predict_logistic,
)
from digihealth_risk.utils.patient_split import apply_canonical_split  # noqa: E402


PHASE8_OUT = ROOT / "digihealth_risk" / "phase_8" / "outputs"
HORIZONS = [1, 2, 3, 4, 5]
HISTORY_WINDOWS = [1, 3, 5]
TRACK = "screening"
FAMILY = "logistic"


def output_paths(no_year: bool) -> tuple[Path, Path, Path]:
    if no_year:
        return (
            PHASE8_OUT / "models_no_year",
            PHASE8_OUT / "model_registry_no_year.json",
            PHASE8_OUT / "deployment_metrics_no_year.csv",
        )
    return (
        PHASE8_OUT / "models",
        PHASE8_OUT / "model_registry.json",
        PHASE8_OUT / "deployment_metrics.csv",
    )


def fit_one(horizon: int, history: int, models_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    key = model_key(TRACK, FAMILY, horizon, history)

    df = load_table(phase0_path(horizon, history))
    df = engineer_features(df)
    train_df, test_df = apply_canonical_split(df)

    artifact = fit_logistic_artifact(train_df)
    artifact["model_family"] = FAMILY
    artifact["track"] = TRACK

    y_train = train_df["Target_AtRisk_Status"].astype(int).to_numpy()
    y_test = test_df["Target_AtRisk_Status"].astype(int).to_numpy()
    train_proba = predict_logistic(artifact, train_df)
    test_proba = predict_logistic(artifact, test_df)
    threshold = float(y_train.mean())

    train_metrics = classification_metrics(y_train, train_proba, threshold)
    test_metrics = classification_metrics(y_test, test_proba, threshold)

    feature_columns = artifact["feature_columns"]
    artifact.update({
        "model_key": key,
        "horizon_years": horizon,
        "history_years": history,
        "threshold": threshold,
        "train_positive_rate": threshold,
        "train_feature_ranges": {
            col: {
                "min": float(train_df[col].min(skipna=True))
                if pd.api.types.is_numeric_dtype(train_df[col]) else None,
                "max": float(train_df[col].max(skipna=True))
                if pd.api.types.is_numeric_dtype(train_df[col]) else None,
            }
            for col in feature_columns
        },
        "intervention_presets": compute_intervention_presets(train_df),
    })

    joblib.dump(artifact, models_dir / f"{key}.joblib")

    metrics_rows = [
        {"model_key": key, "track": TRACK, "model_family": FAMILY,
         "horizon_years": horizon, "history_years": history, "split": "train", **train_metrics},
        {"model_key": key, "track": TRACK, "model_family": FAMILY,
         "horizon_years": horizon, "history_years": history, "split": "test", **test_metrics},
    ]

    registry_entry: dict[str, Any] = {
        "key": key,
        "track": TRACK,
        "model_family": FAMILY,
        "horizon_years": horizon,
        "history_years": history,
        "threshold": round(threshold, 6),
        "test_pr_auc": round(float(test_metrics["pr_auc"]), 4),
        "test_roc_auc": round(float(test_metrics["roc_auc"]), 4),
        "test_brier": round(float(test_metrics["brier"]), 4),
        "feature_count": len(feature_columns),
        "model_path": str((models_dir / f"{key}.joblib").relative_to(ROOT)),
        "intervention_presets": list(artifact["intervention_presets"].keys()),
    }
    return pd.DataFrame(metrics_rows), registry_entry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 8 logistic-only screening export.")
    parser.add_argument(
        "--no-year",
        action="store_true",
        help=(
            "Drop Year, Year_centered, and Year_centered_sq from training "
            "(phase 7 ablation). Writes to models_no_year/."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models_dir, registry_path, metrics_path = output_paths(args.no_year)

    if args.no_year:
        from digihealth_risk.phase_7.year_ablation_utils import patch_drop_year_features
        patch_drop_year_features()
        print("[phase_8] no-Year mode: Year, Year_centered, Year_centered_sq excluded.")

    PHASE8_OUT.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: list[pd.DataFrame] = []
    registry_entries: list[dict] = []

    for history in HISTORY_WINDOWS:
        for horizon in HORIZONS:
            print(f"[phase_8] {TRACK} {FAMILY} N={horizon} M={history} ...", flush=True)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                metrics_df, entry = fit_one(horizon, history, models_dir)
            all_metrics.append(metrics_df)
            registry_entries.append(entry)
            test_row = metrics_df[metrics_df["split"] == "test"].iloc[0]
            print(f"            PR-AUC={test_row['pr_auc']:.4f}  ROC-AUC={test_row['roc_auc']:.4f}")

    pd.concat(all_metrics, ignore_index=True).to_csv(metrics_path, index=False)

    registry = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "variant": "no_year" if args.no_year else "with_year",
        "year_features_excluded": (
            ["Year", "Year_centered", "Year_centered_sq"] if args.no_year else []
        ),
        "horizons": HORIZONS,
        "history_windows": HISTORY_WINDOWS,
        "tracks": {
            "screening": {
                "purpose": "Logistic-only alternative screening track (phase 8).",
                "family_per_horizon": {str(n): FAMILY for n in HORIZONS},
                "rationale": (
                    "Single-family, dependency-light alternative to the phase 6 "
                    "mixed-family screening track. Uses sklearn LogisticRegression "
                    "at every horizon for a uniform serialization profile."
                ),
            },
        },
        "models": registry_entries,
    }
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")

    label = "no_year" if args.no_year else "with_year"
    print(f"\n[phase_8] Exported {len(registry_entries)} models ({label}) -> {models_dir}")
    print(f"[phase_8] Registry  -> {registry_path}")
    print(f"[phase_8] Metrics   -> {metrics_path}")


if __name__ == "__main__":
    main()
