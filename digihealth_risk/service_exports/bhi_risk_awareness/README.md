# IEEE BHI baseline release exporter

This package creates the five-model release consumed by the downstream BHI
**Dysglycemia Risk Awareness** research prototype. It is not a HealthCom
pipeline phase, a new model-family experiment, or a production runtime
dependency.

Run from the modeling repository root:

```bash
python -m digihealth_risk.service_exports.bhi_risk_awareness \
  --output-dir ../DM_risk_prediction/.local/model_releases/bhi-ridge-m5-no-year-v1
```

The command requires the private cohort and existing Phase 0 `M=5` modeling
tables. It fits each horizon on the canonical training-plus-calibration patients,
keeps the canonical test patients untouched, excludes calendar-year features,
and writes five `.npz` files plus `manifest.json`.

The output directory is deliberately external and must remain ignored by Git.
Artifacts contain numeric model parameters and aggregate preprocessing metadata,
not source rows or patient identifiers. Production loads the files with NumPy
`allow_pickle=False` and verifies every SHA-256 digest.
