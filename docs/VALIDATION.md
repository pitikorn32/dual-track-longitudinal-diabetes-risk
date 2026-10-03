# Software validation and known result differences

The software checks below run without patient data. The reference results later
in this document use the private study cohort and cannot be reproduced with the
synthetic example. See [methods](PUBLICATION.md) for data requirements and
analysis commands.

## Check the software

From the repository root, after installing `requirements-dev.txt`:

```bash
python -m pytest -q
python -m digihealth_risk.publication.smoke
bash reproduce.sh --profile healthcom --dry-run
bash reproduce.sh --profile thesis --dry-run
```

The smoke command fits a model to artificial data and checks seven favorable
scenarios. The dry runs list commands without executing the research pipeline.

Tests cover average precision and tied ROC scores, cohort/target alignment,
required model-grid coverage, patient split isolation, future clinical-value
exclusion, exact-artifact safety joins, scenario clipping and derived-feature
refresh, patient-bootstrap multiplicity, and runner argument handling.
Deployment API checks also cover readiness for each variant's complete model
set, removal of stale models during reload, and HTTP 422 responses for
non-finite clinical and questionnaire values. These checks require no trained
artifacts.

## Reference evaluation on the study cohort

The following results cover 28 screening configurations per horizon and 45
monotonic configurations, evaluated on identical test occasions across model
families. They were first checked using saved predictions and subsequently
confirmed by a complete research-workflow rerun from the private source cohort.
They describe the current implementation and dependencies; they do not replace
the published results or establish every historical model's training environment.

### Full research-workflow validation

A fresh Python 3.12 environment on Linux installed `requirements-dev.txt` and
passed `pip check`. In a separate checkout with empty output directories,
`bash reproduce.sh --profile thesis --force --fail-fast` completed **52 steps
with zero failures and zero skipped steps**. The thesis profile includes the
HealthCom workflow, calendar-time ablation, and thesis supplements. Serving
exports are a separate opt-in workflow.

The fresh run reproduced all 15 modeling tables' values and the canonical
patient assignments. All 140 screening rows matched the reference PR-AUC,
ROC-AUC, and Brier values exactly. The intervention evaluation covered 45
configurations and 315 scenario summaries, with zero unexpected increases
across 1,968,057 directional checks. All eight publication figures were
generated. The report recorded fit-time metadata for all 15 statistical
Logistic fits and 30 screening-tree fits, with no missing records in that scope.

All **41 tests passed** with the private cohort and rebuilt modeling tables.
Without the private cohort, the public checks passed with 40 tests and one
cohort-dependent export test skipped; the synthetic example also passed.
PyArrow is included in the requirements because saved pandas modeling tables
can contain Arrow-backed columns.

The supplied cumulative outcome categories agree with the study cutoffs:
non-DM ≤100, pre-DM >100 through 125, and DM >125 mg/dL. Questionnaire collection
dates remain unverified.

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

### XGBoost version sensitivity at the three-year horizon

The paper's screening result is supported by the original prediction files:
XGBoost has AP 0.396444 and ROC-AUC 0.815712, ahead of CatBoost at AP 0.395308.
A controlled refit using the current training code, identical modeling data,
patient assignments, feature values, and model settings reproduces those
XGBoost predictions with version 2.0.3. Changing only XGBoost to version 3.3.0
reproduces the later predictions used in the reference evaluation above.

| XGBoost version | Three-year screening AP | ROC-AUC | Leading family |
| --- | ---: | ---: | --- |
| 2.0.3 | 0.396444 | 0.815712 | XGBoost |
| 3.3.0 | 0.388361 | 0.814769 | CatBoost (AP 0.395308) |

Both fits use 19,286 training occasions, 6,394 calibration occasions, and
6,211 test occasions, with five-year history. Raw and Platt-calibrated
predictions have the same AP. Each refit matches its corresponding saved raw
predictions within `1e-15` probability units. The ranking change therefore
reflects XGBoost version sensitivity, not a misreported screening score or a
change in the evaluation cohort. The current dependency file pins 3.3.0;
reproducing this published fit requires 2.0.3 in a separate environment.
This check covers this screening configuration, not an end-to-end validation
of every phase under 2.0.3.

The three-year monotonic XGBoost fit shows the same version sensitivity. With
the same training code, data, patient split, and feature settings, version
2.0.3 gives AP **0.408668**, reproducing the paper's 0.4087; version 3.3.0 gives
**0.406895**. Each fit reproduces the corresponding saved model's test
probabilities exactly. Both use 25,680 training occasions and 6,211 test
occasions. This resolves both three-year XGBoost score differences without
changing outcome labels or model settings.

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

## Feature-ablation metric definitions

The paper's logistic feature-ablation values, 0.174 to 0.191, are reproduced by
trapezoidal precision-recall integration on the original saved predictions.
The shared leaderboard instead uses average precision. Calculating both
metrics on those same original predictions gives:

| Logistic feature set | Trapezoidal PR area | Average precision |
| --- | ---: | ---: |
| Earlier features | 0.174084 | 0.176704 |
| Enriched features | 0.191282 | 0.193980 |

The improvement remains under either metric, but their numerical values are
not interchangeable. The public evaluation utilities consistently use average
precision. The current paired feature-removal experiment is separately defined
in the methods guide and is not identical to the earlier-versus-enriched
experiment underlying these original predictions.

## Calendar-time aggregation

The original ablation tables support the paper's approximately 0.009 mean
PR-AUC loss for the tree comparisons: 0.009391 for uncalibrated trees and
0.009332 for calibrated trees. The monotonic comparison has a different mean
loss, 0.005944. The paper's horizon-specific changes (about -0.021 at one year
and near zero from three years onward) match the monotonic comparison.
These summaries concern different groups of models and should be identified
separately when reporting an overall mean and horizon-specific changes.

The fresh run with the current dependencies produced mean PR-AUC deltas
(`no_year - baseline`) of -0.009396 for uncalibrated trees, -0.008604 for
calibrated trees, and -0.005999 for monotonic models. These are current-run
results, distinct from the original manuscript-supporting values above.

## Remaining limits

The feature ablation removes the engineered terms listed in the methods guide;
it does not recreate the earlier feature set used for the paper's ablation.
Tree importance describes research refits rather than the original fitted
trees. The fresh calendar-time experiment validates the current implementation;
it does not recreate the historical software environment.
Questionnaire collection dates remain unverified; static lifestyle values are
an explicit research assumption.

The public repository contains code and synthetic examples. The report manifest
records the reporting revision and input hashes; it does not recover missing
information about how earlier models were trained.
