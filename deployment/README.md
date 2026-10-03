# deployment: DigiHealth Risk Score API

Standalone serving implementation of the longitudinal diabetes-risk model: a dual-track
FastAPI service plus the script that trains and exports the model artifacts it
serves. This folder has no imports into the `digihealth_risk/` phase tree; the
modeling helpers it needs are vendored in `modeling.py` and `patient_split.py`,
so it builds and runs on its own.

The exporter refits serving models from the modeling tables using the canonical
patient split. These artifacts and their probabilities have their own metrics;
they are not frozen copies of the HealthCom or thesis benchmark models. At the
five-year screening horizon, the exporter uses Logistic in place of the GEE
winner reported in the research comparison.

## Contents

| File | Purpose |
|------|---------|
| `api.py` | FastAPI app: `/predict`, `/predict/interventions`, `/no_year/*`, `/logistic_only/*` |
| `schemas.py` | Pydantic request/response models (the wire contract) |
| `export_models.py` | Trains and exports the 30 model artifacts |
| `modeling.py` | Vendored feature engineering, preprocessing, monotone rules |
| `patient_split.py` | Vendored canonical 60/20/20 patient split |
| `Dockerfile` | Container image for the API |
| `requirements.txt` | Pinned serving dependencies |

## 1. Export the models

`export_models.py` trains 30 artifacts (2 tracks x 5 horizons x 3 history
windows) from the 15 phase-0 modeling tables.

```bash
pip install -r requirements.txt
python export_models.py
```

By default it reads the modeling tables from the sibling phase tree
(`../digihealth_risk/phase_0/outputs/`). Build them first if they are missing:

```bash
cd ..
for N in 1 2 3 4 5; do for M in 1 3 5; do
  python digihealth_risk/phase_0/build_modeling_tables.py \
    --horizon-years "$N" --history-years "$M"
done; done
```

Override the input location with `DIGIHEALTH_PHASE0_DIR`. Artifacts are written
to this folder: `models/`, `model_registry.json`, `deployment_metrics.csv`.

Optional construct-validity variant (powers the `/no_year/*` routes):

```bash
python export_models.py --no-year
```

Optional logistic-only variant (powers the `/logistic_only/*` routes):

```bash
python export_models.py --logistic-only              # with-Year
python export_models.py --logistic-only --no-year    # no-Year
```

Each `--logistic-only` invocation writes **30 artifacts**: 15 screening
artifacts using sklearn logistic regression at every horizon, plus 15
intervention artifacts using monotonic-constrained logistic regression at
every horizon. Outputs go to `models_logistic_only/` or
`models_logistic_only_no_year/`.

### Why logistic-only?

The default `/predict` route uses a fixed serving-family map (CatBoost at
N=1, XGBoost at N=3, Logistic at N=2/4/5), which means the response
`model_family` field varies by horizon. The `/logistic_only/*` route tree was
added for frontend consumers that want a uniform single-family output for
easier client-side post-processing (e.g. coefficient-based explanations, linear
score decomposition). Compare the exported variants' `deployment_metrics*.csv`
files to assess their performance; manuscript leaderboard scores describe the
research fits.

Intervention is also exposed under `/logistic_only/*` via
**monotonic-constrained logistic regression**: coefficient sign bounds enforced
at fit time specify directional relationships between features and the score.
The closed-form sigmoid prediction path is identical to the unconstrained
logistic; only the fit differs. The research scenario checks cover the study
cohort and specified presets, rather than every possible API request.

Presets clip proposed values to the training range, then preserve the favorable
direction specified by the exporter's monotonic constraints. If a training
bound would reverse the change, the current value is retained: BMI 17 stays 17
when the training minimum is 18, and exercise above the training maximum is
never reduced by an "increase exercise" preset. Training coverage still limits
the validity of predictions outside the observed range.

Missing intervention values remain missing and are skipped by the preset;
the fitted preprocessor handles them identically for baseline and scenario
scoring. Other observed features can still change in a combined preset.
Returned `changed_features` lists the actual values used, including equal
`from`/`to` values when a proposed update is blocked; skipped missing features
are omitted. These guards apply to existing exported models without retraining.
Directional consistency does not establish a causal treatment effect.

## 2. Run the API

```bash
uvicorn api:app --reload --port 8000
```

Interactive docs: http://localhost:8000/docs

All four health endpoints return HTTP 200 only when their own 30 expected
models are loaded. An empty or incomplete model set returns HTTP 503 with
`status` (`models_not_loaded` or `models_incomplete`) and `missing_model_keys`.
Optional variants report their readiness independently of the default variant.

Numeric measurement, questionnaire, and cumulative-history inputs must be
finite numbers or null. NaN, infinity, and overflowing values are rejected
with HTTP 422. Validation responses include each error's `type`, `loc`, and
`msg`; invalid input values are omitted from those responses.

From the repository root, the public regression checks run with
`python -m pytest tests/deployment -q` after installing `requirements-dev.txt`.
They use synthetic inputs and isolated registries to check complete and partial
model sets, model reloads, finite-number validation, and favorable preset
directions for inputs within and outside training bounds across all prediction
routes. They also verify that missing intervention values remain missing.
They do not require private data or trained artifacts.

## 3. Docker

```bash
# Export the artifacts first so they are baked into the image:
python export_models.py

docker build -t digihealth-risk-api .
docker run -p 8000:8000 digihealth-risk-api

# Or keep the image artifact-free and serve models from the host:
docker run -p 8000:8000 -v "$(pwd)/models:/app/models" digihealth-risk-api
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Loaded-model status per track |
| GET | `/models` | List loaded artifacts |
| GET | `/models/{key}` | One artifact's metadata |
| POST | `/predict` | Passive-screening risk score (mixed family per horizon) |
| POST | `/predict/interventions` | What-if simulation using monotonic models |
| GET / POST | `/no_year/*` | The same route tree with Year features excluded |
| POST | `/logistic_only/predict` | Screening using logistic at every horizon (uniform single-family output) |
| POST | `/logistic_only/predict/interventions` | Intervention using monotonic-constrained logistic at every horizon |
| POST | `/logistic_only/no_year/predict` | Logistic-only screening, Year features excluded |
| POST | `/logistic_only/no_year/predict/interventions` | Monotonic-logistic intervention, Year features excluded |
| GET | `/logistic_only/health`, `/logistic_only/models`, `/logistic_only/models/{key}` | Logistic-only registry surface |
| GET | `/logistic_only/no_year/health`, `/logistic_only/no_year/models`, `/logistic_only/no_year/models/{key}` | Logistic-only no-Year registry surface |

Prediction and individual-model routes return 404 when their requested artifact
is absent. Export the default models before using `/predict` or
`/predict/interventions`, and export each optional variant to enable its
predictions. Model-list routes return the loaded artifacts, including an empty
list when none are available; health routes report incomplete sets with 503.

## Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `DIGIHEALTH_MODEL_DIR` | `./models` | With-Year mixed-family artifacts |
| `DIGIHEALTH_MODEL_DIR_NO_YEAR` | `./models_no_year` | No-Year mixed-family artifacts |
| `DIGIHEALTH_MODEL_DIR_LOGISTIC_ONLY` | `./models_logistic_only` | With-Year logistic-only artifacts |
| `DIGIHEALTH_MODEL_DIR_LOGISTIC_ONLY_NO_YEAR` | `./models_logistic_only_no_year` | No-Year logistic-only artifacts |
| `DIGIHEALTH_PHASE0_DIR` | `../digihealth_risk/phase_0/outputs` | Modeling tables for export |
| `DIGIHEALTH_DATA` | `../datasets/longitudinal_cohort.pkl` | Source cohort for the split |
| `DIGIHEALTH_SPLIT_CACHE` | `../digihealth_risk/phase_0/outputs/patient_split.csv` | Canonical split cache |
