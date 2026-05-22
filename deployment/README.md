# deployment: DigiHealth Risk Score API

Standalone production slice of the longitudinal diabetes-risk model: a dual-track
FastAPI service plus the script that trains and exports the model artifacts it
serves. This folder has no imports into the `digihealth_risk/` phase tree; the
modeling helpers it needs are vendored in `modeling.py` and `patient_split.py`,
so it builds and runs on its own.

## Contents

| File | Purpose |
|------|---------|
| `api.py` | FastAPI app: `/predict`, `/predict/interventions`, `/no_year/*` |
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

## 2. Run the API

```bash
uvicorn api:app --reload --port 8000
```

Interactive docs: http://localhost:8000/docs

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
| POST | `/predict` | Passive-screening risk score |
| POST | `/predict/interventions` | Intervention-safe what-if simulation |
| GET / POST | `/no_year/*` | The same route tree with Year features excluded |

The `/no_year/*` routes return 404 until `export_models.py --no-year` has been
run; `/predict` and `/predict/interventions` work regardless.

## Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `DIGIHEALTH_MODEL_DIR` | `./models` | With-Year artifacts the API loads |
| `DIGIHEALTH_MODEL_DIR_NO_YEAR` | `./models_no_year` | No-Year artifacts |
| `DIGIHEALTH_PHASE0_DIR` | `../digihealth_risk/phase_0/outputs` | Modeling tables for export |
| `DIGIHEALTH_DATA` | `../datasets/df_final.pkl` | Source cohort for the split |
| `DIGIHEALTH_SPLIT_CACHE` | `../digihealth_risk/phase_0/outputs/patient_split.csv` | Canonical split cache |
