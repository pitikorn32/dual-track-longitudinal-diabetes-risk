from __future__ import annotations

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
