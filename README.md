# dual-track-longitudinal-diabetes-risk

Production implementation of a longitudinal diabetes-risk prediction model that
predicts diabetes risk from 12-year Thai healthcare data (6,892 patients,
2005-2016).

## Layout

```
longitudinal-diabetes-risk/
├─ digihealth_risk/    modeling pipeline, phases 0-7 (training + research)
├─ deployment/         standalone dual-track FastAPI serving slice
├─ reproduce.sh        runs the full phase pipeline in dependency order
└─ requirements.txt    pinned dependencies for the phase pipeline
```

The repository has two parts, each with its own README:

- **`digihealth_risk/`** is the modeling pipeline: data engineering, statistical
  models, tree models, survival models, calibration, monotonic intervention
  models, and a deployment phase, organised as phase scripts. See
  `digihealth_risk/README.md` for the full run order.
- **`deployment/`** is the standalone production slice: a dual-track FastAPI
  service (`/predict`, `/predict/interventions`) plus the script that exports
  its model artifacts. It is self-contained and has no imports into the phase
  tree. See `deployment/README.md`.

## Data

The source cohort (`datasets/df_final.pkl`, roughly 5.6 MB of de-identified
patient data) is not committed. Place it at `datasets/df_final.pkl` before
running the pipeline. Generated artifacts (modeling tables, model files, logs)
are git-ignored.

## Quick start

```bash
pip install -r requirements.txt
```

Option A: run the entire phase pipeline (training and research, takes hours).

```bash
bash reproduce.sh
```

Option B: build only the 15 modeling tables the deployment export needs.

```bash
for N in 1 2 3 4 5; do for M in 1 3 5; do
  python digihealth_risk/phase_0/build_modeling_tables.py \
    --horizon-years "$N" --history-years "$M"
done; done
```

Then export the serving artifacts and start the API:

```bash
cd deployment
pip install -r requirements.txt
python export_models.py
uvicorn api:app --port 8000
```

Interactive API docs: http://localhost:8000/docs

## Conventions

- **Canonical patient split**: 60/20/20 by `PatientId`, seed `20260501`. The
  phase tree and the deployment slice share the same split.
- **Primary metric**: PR-AUC (handles class imbalance). Secondary: ROC-AUC,
  Brier score.

## Thesis

This pipeline accompanies the thesis *Longitudinal Diabetes Risk Prediction from
12-Year Thai Healthcare Data*. Full methodology and results, including the
section numbers cited throughout these READMEs, are in the thesis PDF:

[Thesis PDF](THESIS_PDF_URL)

<!-- TODO: replace THESIS_PDF_URL above with the public link to the thesis PDF -->

The thesis figure-generation scripts are not included; every phase script that
produces a modeling result is.
