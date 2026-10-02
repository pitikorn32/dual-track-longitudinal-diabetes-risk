# Dual track longitudinal diabetes risk

Research code accompanying the IEEE HealthCom paper **Dual-Track Longitudinal
Modeling of Diabetes Risk: Horizon-Specific Screening and Intervention-Safe
Scoring**, with supplementary analyses for the extended thesis.

The study compares screening predictions and monotonic what-if scores across
five prediction horizons and three history windows. HealthCom is the primary
workflow; thesis supplements and serving variants are identified separately.

The private patient cohort is **not distributed**. This repository provides the
methods, input schema, executable analyses, and an entirely synthetic example.
The example verifies software behavior; it cannot reproduce the paper's numbers.

## Start without private data

Use Python 3.12 and install the pinned dependencies:

```bash
pip install -r requirements-dev.txt
python -m digihealth_risk.publication.smoke
python -m pytest tests -q
bash reproduce.sh --profile healthcom --dry-run
```

The pinned XGBoost version is 3.3.0. The paper's three-year screening fit is
reproduced with XGBoost 2.0.3; the version changes its score and rank. See the
[version comparison](docs/VALIDATION.md#xgboost-version-sensitivity-at-the-three-year-horizon)
before comparing results with the paper.

The smoke command constructs an artificial cohort in a temporary directory,
builds rolling features, uses the canonical patient split, fits monotonic
XGBoost, and checks all seven favorable scenarios. It does not read private data
or alter the study split cache.

## Run the research with authorized data

Place a trusted cohort file at `datasets/df_final.pkl`, following the
[data schema and methods](docs/PUBLICATION.md). Run from this repository root:

```bash
bash reproduce.sh --profile healthcom --fail-fast
bash reproduce.sh --profile thesis --fail-fast  # HealthCom plus thesis supplements
bash reproduce.sh --profile healthcom --from-phase 4 --fail-fast
```

These runs can take hours. Existing statistical-grid results are reused unless
`--force` is supplied; other stages may overwrite outputs. Preserve previous
outputs before rerunning. `--from-phase` expects earlier results to exist.
`--no-ablation` deliberately skips the calendar-time experiment and therefore
produces an incomplete publication run. Deployment export is opt-in through
`--with-deploy`.

For existing trained artifacts and predictions, reevaluate without retraining:

```bash
python digihealth_risk/phase_4/cross_family_comparison.py
python digihealth_risk/phase_4/bootstrap_significance.py
python digihealth_risk/phase_5/evaluate_saved_models.py
python digihealth_risk/phase_5/intervention_benchmark.py
python -m digihealth_risk.publication.report
```

Generated patient-level predictions, modeling tables, models, logs, and figures
remain ignored by Git. Reporting writes aggregate figures and a provenance
manifest under `digihealth_risk/publication/outputs/`.

## Publication contract

- Patient-grouped 60/20/20 train/calibration/test assignment, seed `20260501`;
  stages without calibration fold calibration patients into training.
- Study-defined first onset: cumulative maximum FBS **>100 mg/dL** is at risk;
  currently at-risk and post-onset rows are excluded.
- Primary metric: **average precision**, called PR-AUC in the shared leaderboard.
- Retrospective comparison of 28 configurations per horizon on identical test
  occasions. Winners are selected from test metrics, not independently validated
  model selection.
- Directional what-if checks establish consistency under specified scenarios;
  they do not estimate causal treatment effects.
- Lifestyle questionnaire values are assumed time invariant; collection timing
  and availability at each source year are unverified.

See [methods and claim-to-command mapping](docs/PUBLICATION.md),
[validation and known differences](docs/VALIDATION.md), and the
[glossary](GLOSSARY.md) for definitions, implementation details, and known
differences from the published results.

## Repository layout

| Path | Purpose |
| --- | --- |
| `digihealth_risk/phase_0` through `phase_7` | Research engineering, fitting, evaluation, and ablations |
| `digihealth_risk/publication/` | Synthetic example, current feature ablation, research importance, and reporting |
| `digihealth_risk/utils/` | Canonical patient split and evaluation checks |
| `tests/` | Scientific invariants, command-line behavior, and BHI export compatibility |
| `deployment/` | Separately trained serving variants; see its README |

The serving API retrains variants and substitutes some families; its outputs
are not frozen copies of the paper's benchmark fits. The BHI exporter under
`digihealth_risk/service_exports/` is a separate downstream application.
