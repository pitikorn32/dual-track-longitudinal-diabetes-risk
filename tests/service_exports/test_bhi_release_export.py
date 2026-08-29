from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "datasets" / "df_final.pkl"
PHASE0 = ROOT / "digihealth_risk" / "phase_0" / "outputs"


@pytest.mark.skipif(not DATA.exists(), reason="private cohort is not available")
def test_exported_release_reproduces_the_evidence_scorer(tmp_path: Path) -> None:
    from digihealth_risk.service_exports.bhi_risk_awareness import (
        export_release,
        load_horizon_artifact,
        predict_portable,
    )

    result = export_release(
        output_dir=tmp_path,
        phase0_dir=PHASE0,
        source_data=DATA,
        release_id="bhi-ridge-m5-no-year-test",
    )

    assert result.manifest_path == tmp_path / "manifest.json"
    assert result.manifest_path.is_file()
    assert sorted(path.name for path in result.artifact_paths) == [
        "horizon_1.npz",
        "horizon_2.npz",
        "horizon_3.npz",
        "horizon_4.npz",
        "horizon_5.npz",
    ]

    for horizon, reference in result.reference_predictions.items():
        artifact = load_horizon_artifact(result.manifest_path, horizon)
        portable = predict_portable(artifact, result.test_frames[horizon])
        np.testing.assert_allclose(portable, reference, rtol=0.0, atol=1e-12)

    manifest_text = result.manifest_path.read_text(encoding="utf-8")
    assert "PatientId" not in manifest_text
    assert "bhi-ridge-m5-no-year-test" in manifest_text
    manifest = json.loads(manifest_text)
    assert manifest["source"]["git_commit"]
    assert manifest["runtime_versions"]["numpy"]
    assert manifest["runtime_versions"]["scikit_learn"]
    assert {
        item["horizon_years"]: item["golden_prediction"]
        for item in manifest["horizons"]
    } == pytest.approx(
        {
            1: 0.013152730258756179,
            2: 0.030432624427624292,
            3: 0.058741939670585851,
            4: 0.083383788869112338,
            5: 0.11369680661767689,
        },
        rel=0.0,
        abs=1e-15,
    )
