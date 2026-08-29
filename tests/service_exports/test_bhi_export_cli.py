from __future__ import annotations

import subprocess
import sys


def test_exporter_exposes_a_documented_command() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "digihealth_risk.service_exports.bhi_risk_awareness",
            "--help",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "--output-dir" in completed.stdout
    assert "IEEE BHI" in completed.stdout
