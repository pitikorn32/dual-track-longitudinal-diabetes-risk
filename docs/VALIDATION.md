# Software validation and known result differences

Validated on 2026-10-02 in the existing `digihealth` conda environment,
Python 3.12.13. The pinned requirements describe this tested environment;
installation into a fresh environment was not tested. Publication references
are the HealthCom v6 camera-ready paper and thesis V7.

## Scope of execution

This validation reevaluated saved research predictions and trained artifacts,
ran targeted research refits, and exercised the synthetic workflow. It was
**not a fresh end-to-end retraining of every phase**.
Saved predictions do not have complete historical training provenance.

| Check | Result |
| --- | --- |
| `python -m pytest -q` | 34 passed, including BHI export compatibility |
| `python -m digihealth_risk.publication.smoke` | Artificial cohort, isolated split cache, monotonic fit, and seven scenarios completed |
| Both runner profiles with `--dry-run`; `bash -n reproduce.sh` | Passed |
| `phase_4/cross_family_comparison.py` | Complete 28-configuration grid at all five horizons; 140 ranking rows |
| `phase_4/bootstrap_significance.py` | 2,000 valid patient-cluster replicates for each of ten comparisons |
| Bootstrap with `--survival-reference best` | Separate sensitivity analysis completed with 2,000 valid replicates per comparison |
| `phase_5/evaluate_saved_models.py` | All 45 configurations evaluated on exact shared rows |
| `phase_5/intervention_benchmark.py` | Performance and directional evidence joined by configuration, artifact, and cohort |
| `publication.feature_ablation` | Eight paired current-fit results across four families |
| `publication.feature_importance` | Research tree refits and saved statistical coefficients for all five horizon leaders |
| `phase_4/feature_effects_analysis.py` | Current effect summaries regenerated |
| `phase_1/compare_statistical_grid.py` | Supplementary comparison regenerated |
| `phase_0/build_cohort_figure.py` | Cohort accounting regenerated; holdout allocation separated from alignment exclusions |
| `publication.report` | Eight PNG/PDF figures, top-five table, and provenance manifest generated; no optional figures omitted |

Module names beginning with `publication.` are run with
`python -m digihealth_risk.publication.<name>`. Script paths begin at
`digihealth_risk/`. See [methods](PUBLICATION.md) for ordered workflows.

Tests cover average precision and tied ROC scores, cohort/target alignment,
required model-grid coverage, patient split isolation, future clinical-value
exclusion, exact-artifact safety joins, scenario clipping and derived-feature
refresh, patient-bootstrap multiplicity, and runner argument handling.

The source-label audit found no disagreement between supplied cumulative
categories and the published study cutoffs (non-DM ≤100, pre-DM >100 through
125, DM >125 mg/dL). This does not establish
prospective availability of the questionnaire fields.

## Current shared-cohort evidence

These aggregate results describe local saved-artifact reevaluation. They are
not new estimates from public data and are not substitutes for the printed
camera-ready table.

| Horizon | Shared test occasions | Screening leader | Screening AP | Monotonic leader | Monotonic AP |
| --- | ---: | --- | ---: | --- | ---: |
| 1 | 8,248 | CatBoost | 0.208537 | EBM | 0.217019 |
| 2 | 7,209 | Logistic | 0.316895 | CatBoost | 0.332610 |
| 3 | 6,211 | CatBoost | 0.395308 | XGBoost | 0.406895 |
| 4 | 5,246 | Logistic | 0.472660 | CatBoost | 0.475754 |
| 5 | 4,325 | GEE | 0.528168 | CatBoost | 0.522601 |

Every listed leader uses five-year history. The seven favorable presets across
45 configurations produced 315 scenario summaries and **1,968,057 directional
checks**, with zero increases above `1e-10` score points. These are repeated
model/scenario evaluations, not independent patients. Passing these presets
does not prove unrestricted monotonicity or causal treatment benefit.

At horizon 3, the camera-ready screening leader is XGBoost (AP 0.3964), while
the available saved predictions select CatBoost (AP 0.395308). The saved
monotonic XGBoost result is 0.406895 versus 0.4087 in the paper. Historical
artifact/software differences remain unresolved; the current environment uses
XGBoost 3.3.0 rather than the former requirements pin of 2.0.3. A version
difference alone does not establish the cause.

## Bootstrap interpretation

Tree-versus-statistical AP-difference intervals include zero at all horizons.
The historical fixed M=5 two-stage survival reference gives winner-minus-
survival intervals excluding zero at horizons 4 and 5, but not 1–3.

The actual best-survival sensitivity changes the four-year conclusion:

| Horizon | Survival reference | AP-difference 95% interval |
| --- | --- | --- |
| 4 | Fixed two-stage M=5 | [0.000271, 0.057735] |
| 4 | Best survival (two-stage M=3) | [-0.002916, 0.049534] |
| 5 | Fixed two-stage M=5 | [0.003171, 0.058729] |
| 5 | Best survival (two-stage M=3) | [0.001330, 0.052826] |

The default command preserves the implemented historical reference and labels
it explicitly. Neither comparison supports claiming that the four-year leader
reliably outperforms every survival configuration. These intervals also do not
adjust for retrospective model selection.

## Remaining limits

The current feature ablation is a defined removal of engineered terms, not a
reconstruction of historical v1 code. Tree importance comes from named research
refits, not recovered frozen historical trees. Previously saved calendar-time
ablation results were plotted, not fully retrained during this validation.
Questionnaire collection dates remain unverified; static lifestyle values are
an explicit research assumption.

Private data, identifiers, predictions, serialized models, figures, and local
snapshots remain ignored by Git. The report manifest records the reporting
revision and input hashes; historical training provenance remains incomplete.
