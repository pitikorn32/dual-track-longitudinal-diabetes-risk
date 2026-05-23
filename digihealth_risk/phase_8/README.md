# Phase 8 — Logistic-only Alternative Screening Build

## Purpose

An **alternative deployment build of the screening track** that uses logistic
regression at every horizon, in both with-Year and no-Year variants. Produces
**30 model artifacts** (15 per variant) with the same `.joblib` schema as the
phase 6 export, so wiring this set into the FastAPI later is a routing change
rather than a re-engineering job.

Why this exists alongside phase 6:

- The phase 6 screening track uses a mixed-family per-horizon winner
  (CatBoost at N=1, XGBoost at N=3, Logistic at N=2/4/5). It ships with three
  runtime dependencies (`catboost`, `xgboost`, `interpret`).
- Phase 8 collapses the family choice to logistic regression at every horizon,
  giving a uniform, dependency-light, fully-reproducible artifact set.
- The intervention track is **not covered** by phase 8.

## Thesis reference

Not a thesis result. Phase 8 is a post-thesis engineering alternative to the
phase 6 deployment, motivated by §6.4 (dual-track deployment) and the
N=5 substitution note that already chose Logistic over GEE for deployability.

## Scripts

| Script | Role |
|---|---|
| `export_logistic_models.py` | Trains 15 screening-logistic artifacts; `--no-year` switches to the phase 7 ablation variant |
| `run_all.sh` | Runs the script twice (with-Year, then no-Year), producing all 30 artifacts |

## Prerequisites

- All 15 phase 0 modeling tables for `N ∈ {1..5} × M ∈ {1, 3, 5}`:
  ```
  digihealth_risk/phase_0/outputs/phase_0_modeling_table_horizon_{N}_history_{M}.pkl
  ```
- Python deps: `pandas numpy scikit-learn scipy joblib` (the existing pipeline requirements). No catboost/xgboost/interpret needed for phase 8 itself.

## Step-by-step

### Run all 30 (both variants)

```bash
bash digihealth_risk/phase_8/run_all.sh
```

### Run one variant only

```bash
# With-Year only (15 artifacts)
python digihealth_risk/phase_8/export_logistic_models.py

# No-Year only (15 artifacts, phase 7 ablation)
python digihealth_risk/phase_8/export_logistic_models.py --no-year
```

### Skip-flag examples

```bash
bash digihealth_risk/phase_8/run_all.sh --skip-no-year      # with-Year only
bash digihealth_risk/phase_8/run_all.sh --skip-with-year    # no-Year only
bash digihealth_risk/phase_8/run_all.sh --fail-fast         # halt on first error
```

## Outputs

All written under `digihealth_risk/phase_8/outputs/` (git-ignored):

| Path | Variant | Count |
|---|---|---|
| `models/screening_logistic_n{N}_m{M}.joblib` | with-Year | 15 |
| `model_registry.json` | with-Year | 1 |
| `deployment_metrics.csv` | with-Year | 1 |
| `models_no_year/screening_logistic_n{N}_m{M}.joblib` | no-Year | 15 |
| `model_registry_no_year.json` | no-Year | 1 |
| `deployment_metrics_no_year.csv` | no-Year | 1 |
| `logs/with_year.log`, `logs/no_year.log` | tee'd stdout | 2 |

## Artifact schema

Each `.joblib` is a dict with the same keys as the phase 6 logistic artifact:

| Key | Description |
|---|---|
| `model_key` | e.g. `screening_logistic_n3_m5` |
| `model_family` | `"logistic"` |
| `track` | `"screening"` |
| `horizon_years`, `history_years` | int |
| `preprocessor` | sklearn `ColumnTransformer` (numeric impute + scale, categorical impute + one-hot) |
| `feature_columns`, `numeric_features`, `categorical_features` | input feature lists |
| `transformed_feature_names` | post-preprocessor names |
| `coefficients` | `[intercept, *coefs]`, length = 1 + transformed feature count |
| `mean_`, `scale_` | per-column standardization stats (post-preprocessor) |
| `threshold` | training positive rate, used as default decision threshold |
| `train_feature_ranges` | per-feature train min/max for downstream bounds checking |
| `intervention_presets` | preset metadata, kept for schema compatibility with phase 6 (screening does not use them) |

Scoring is closed-form:
`p = sigmoid(coefficients · [1, scale(preprocessor.transform(x))])`.

See `digihealth_risk/phase_6/export_models.py::predict_logistic` for the
canonical scoring function (phase 8 artifacts are drop-in compatible).

## Expected runtime

~3–8 minutes per variant on a standard laptop. The full
`run_all.sh` finishes in well under 20 minutes.

## Not in scope

- API wiring. Phase 8 only writes artifacts and a registry. Adding a
  `/logistic_only/*` route tree to `digihealth_risk/phase_6/api.py` is a
  follow-up change.
- Calibration. Phase 4-style Platt/isotonic is not applied; sklearn lbfgs
  logistic output is used as-is.
- Intervention track. Screening only.
- `reproduce.sh`. Phase 8 is an alternative build, not part of the canonical
  thesis reproduction. Add `bash digihealth_risk/phase_8/run_all.sh` to
  `reproduce.sh` only if you want the alternative set produced by every full
  reproduction.
