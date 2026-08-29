"""Export the frozen IEEE BHI baseline scorer as a portable release bundle.

This downstream packaging module is deliberately separate from the HealthCom
model-family pipeline.  Its public interface is ``export_release``; loading and
portable prediction are exposed so a consumer can verify the generated bundle
without deserializing executable Python objects.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


SUBMODULE_ROOT = Path(__file__).resolve().parents[3]
DEPLOYMENT = SUBMODULE_ROOT / "deployment"
if str(DEPLOYMENT) not in sys.path:
    sys.path.insert(0, str(DEPLOYMENT))

import export_models as legacy_export  # noqa: E402
import modeling  # noqa: E402
import patient_split  # noqa: E402


HORIZONS = (1, 2, 3, 4, 5)
HISTORY_YEARS = 5
RIDGE_ALPHA = 0.01


@dataclass(frozen=True)
class PortableArtifact:
    coefficients: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    numeric_imputation: np.ndarray
    numeric_features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    categorical_imputation: tuple[Any, ...]
    categorical_categories: tuple[tuple[Any, ...], ...]


@dataclass(frozen=True)
class ExportResult:
    manifest_path: Path
    artifact_paths: tuple[Path, ...]
    reference_predictions: dict[int, np.ndarray]
    test_frames: dict[int, pd.DataFrame]


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if pd.isna(value):
        return None
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_metadata(artifact: dict[str, Any]) -> dict[str, Any]:
    preprocessor = artifact["preprocessor"]
    numeric_imputer = preprocessor.named_transformers_["num"]
    categorical_pipeline = preprocessor.named_transformers_["cat"]
    categorical_imputer = categorical_pipeline.named_steps["imputer"]
    encoder = categorical_pipeline.named_steps["onehot"]
    return {
        "numeric_features": list(artifact["numeric_features"]),
        "categorical_features": list(artifact["categorical_features"]),
        "categorical_imputation": [
            _json_value(value) for value in categorical_imputer.statistics_
        ],
        "categorical_categories": [
            [_json_value(value) for value in category]
            for category in encoder.categories_
        ],
        "transformed_feature_names": list(artifact["transformed_feature_names"]),
        "numeric_imputation": np.asarray(
            numeric_imputer.statistics_, dtype=float
        ),
    }


def _write_horizon_artifact(
    path: Path,
    artifact: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    np.savez_compressed(
        path,
        coefficients=np.asarray(artifact["coefficients"], dtype=np.float64),
        mean=np.asarray(artifact["mean_"], dtype=np.float64),
        scale=np.asarray(artifact["scale_"], dtype=np.float64),
        numeric_imputation=np.asarray(metadata["numeric_imputation"], dtype=np.float64),
    )


def _configure_sources(phase0_dir: Path, source_data: Path) -> None:
    patient_split.SOURCE_DATA = source_data
    patient_split.SPLIT_CACHE = phase0_dir / "patient_split.csv"


def export_release(
    *,
    output_dir: Path,
    phase0_dir: Path,
    source_data: Path,
    release_id: str,
) -> ExportResult:
    """Fit, export, and verify the five evidence-producing BHI scorers.

    The caller owns ``output_dir``. Existing release files are rejected so an
    immutable release cannot be silently overwritten.
    """

    output_dir = Path(output_dir)
    phase0_dir = Path(phase0_dir)
    source_data = Path(source_data)
    if not source_data.is_file():
        raise FileNotFoundError(f"Private source cohort not found: {source_data}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Release directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    _configure_sources(phase0_dir, source_data)
    modeling.patch_drop_year_features()

    entries: list[dict[str, Any]] = []
    paths: list[Path] = []
    references: dict[int, np.ndarray] = {}
    test_frames: dict[int, pd.DataFrame] = {}

    for horizon in HORIZONS:
        table_path = phase0_dir / (
            f"phase_0_modeling_table_horizon_{horizon}_history_{HISTORY_YEARS}.pkl"
        )
        table = modeling.engineer_features(modeling.load_table(table_path))
        train, calibration, test = patient_split.apply_canonical_split(
            table, return_calibration=True
        )
        fit = pd.concat([train, calibration], ignore_index=True)
        artifact = legacy_export.fit_logistic_artifact(fit)
        if any("Year" in name for name in artifact["feature_columns"]):
            raise AssertionError("Calendar-year feature leaked into BHI artifact")

        reference = legacy_export.predict_logistic(artifact, test)
        metadata = _artifact_metadata(artifact)
        artifact_path = output_dir / f"horizon_{horizon}.npz"
        _write_horizon_artifact(artifact_path, artifact, metadata)

        y_test = test["Target_AtRisk_Status"].astype(int).to_numpy()
        entry = {
            "horizon_years": horizon,
            "artifact": artifact_path.name,
            "artifact_sha256": _sha256(artifact_path),
            "feature_columns": list(artifact["feature_columns"]),
            "numeric_features": metadata["numeric_features"],
            "categorical_features": metadata["categorical_features"],
            "categorical_imputation": metadata["categorical_imputation"],
            "categorical_categories": metadata["categorical_categories"],
            "transformed_feature_names": metadata["transformed_feature_names"],
            "fit_rows": int(len(fit)),
            "fit_patients": int(fit["PatientId"].nunique()),
            "test_rows": int(len(test)),
            "test_patients": int(test["PatientId"].nunique()),
            "test_events": int(y_test.sum()),
            "test_metrics": {
                "roc_auc": float(roc_auc_score(y_test, reference)),
                "pr_auc": float(average_precision_score(y_test, reference)),
                "brier": float(brier_score_loss(y_test, reference)),
            },
        }
        entries.append(entry)
        paths.append(artifact_path)
        references[horizon] = reference
        test_frames[horizon] = test

    manifest = {
        "schema_version": 1,
        "release_id": release_id,
        "outcome": "first transition of the source AtRisk label from 0 to 1",
        "model_family": "raw ridge logistic regression",
        "ridge_alpha": RIDGE_ALPHA,
        "sklearn_c": 1.0 / RIDGE_ALPHA,
        "history_years": HISTORY_YEARS,
        "calendar_year_features": False,
        "post_hoc_calibration": False,
        "split": "canonical 60/20/20 patient split; train and calibration folded",
        "horizons": entries,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return ExportResult(
        manifest_path=manifest_path,
        artifact_paths=tuple(paths),
        reference_predictions=references,
        test_frames=test_frames,
    )


def load_horizon_artifact(manifest_path: Path, horizon: int) -> PortableArtifact:
    """Load and integrity-check one non-executable horizon artifact."""

    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    try:
        entry = next(
            item for item in manifest["horizons"] if item["horizon_years"] == horizon
        )
    except StopIteration as exc:
        raise KeyError(f"Horizon {horizon} is absent from {manifest_path}") from exc
    artifact_path = manifest_path.parent / entry["artifact"]
    if _sha256(artifact_path) != entry["artifact_sha256"]:
        raise ValueError(f"Artifact digest mismatch: {artifact_path.name}")
    with np.load(artifact_path, allow_pickle=False) as arrays:
        return PortableArtifact(
            coefficients=arrays["coefficients"].copy(),
            mean=arrays["mean"].copy(),
            scale=arrays["scale"].copy(),
            numeric_imputation=arrays["numeric_imputation"].copy(),
            numeric_features=tuple(entry["numeric_features"]),
            categorical_features=tuple(entry["categorical_features"]),
            categorical_imputation=tuple(entry["categorical_imputation"]),
            categorical_categories=tuple(
                tuple(values) for values in entry["categorical_categories"]
            ),
        )


def predict_portable(artifact: PortableArtifact, frame: pd.DataFrame) -> np.ndarray:
    """Score a model-table frame using only exported arrays and metadata."""

    numeric = frame.loc[:, artifact.numeric_features].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=float)
    missing_numeric = np.isnan(numeric)
    if missing_numeric.any():
        numeric[missing_numeric] = np.take(
            artifact.numeric_imputation, np.nonzero(missing_numeric)[1]
        )

    encoded_parts: list[np.ndarray] = []
    for feature, imputation, categories in zip(
        artifact.categorical_features,
        artifact.categorical_imputation,
        artifact.categorical_categories,
        strict=True,
    ):
        values = frame[feature].astype("object").where(frame[feature].notna(), imputation)
        encoded_parts.append(
            np.column_stack([(values == category).to_numpy(dtype=float) for category in categories])
        )
    raw = np.hstack([numeric, *encoded_parts]) if encoded_parts else numeric
    scaled = (raw - artifact.mean) / artifact.scale
    design = np.hstack([np.ones((len(frame), 1), dtype=float), scaled])
    eta = design @ artifact.coefficients
    return np.where(eta >= 0, 1.0 / (1.0 + np.exp(-eta)), np.exp(eta) / (1.0 + np.exp(eta)))
