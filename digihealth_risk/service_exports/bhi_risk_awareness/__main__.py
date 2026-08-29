"""Command-line entry point for the downstream IEEE BHI model exporter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import SUBMODULE_ROOT, export_release


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export the five frozen IEEE BHI risk-awareness baseline models. "
            "This is downstream service packaging, not a HealthCom pipeline phase."
        )
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--phase0-dir",
        type=Path,
        default=SUBMODULE_ROOT / "digihealth_risk" / "phase_0" / "outputs",
    )
    parser.add_argument(
        "--source-data",
        type=Path,
        default=SUBMODULE_ROOT / "datasets" / "df_final.pkl",
    )
    parser.add_argument(
        "--release-id",
        default="bhi-ridge-m5-no-year-v1",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = export_release(
        output_dir=args.output_dir,
        phase0_dir=args.phase0_dir,
        source_data=args.source_data,
        release_id=args.release_id,
    )
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    summary = {
        "release_id": manifest["release_id"],
        "manifest": str(result.manifest_path),
        "artifacts": {
            str(item["horizon_years"]): item["artifact_sha256"]
            for item in manifest["horizons"]
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
