# Longitudinal Diabetes Risk Study

Shared terminology for the HealthCom camera-ready paper and the extended thesis. These definitions describe their common study concepts.

## Language

**Source year**:
The calendar year at which a patient's eligible prediction is made and clinical history ends, denoted by T.

**Prediction horizon**:
The number of calendar years from the source year to the target year, denoted by N; the study evaluates one through five years.
_Avoid_: Follow-up duration when referring specifically to N

**History window**:
The span of calendar years summarized for a prediction, ending in and including the source year, denoted by M. The study evaluates one, three, and five years, with shorter available histories near the beginning of the study period.
_Avoid_: Number of visits

**Rolling patient-year**:
A prediction occasion for one patient at one eligible source year; a patient may contribute multiple occasions before first onset.

**Static lifestyle assumption**:
The study's treatment of one set of patient questionnaire responses as unchanged across source years. This assumption does not establish when the responses were collected or whether they were available at each prediction occasion.
_Avoid_: Verified baseline questionnaire

**First onset**:
The first recorded transition from the study's non-at-risk state to its combined pre-diabetes or diabetes state, derived from cumulative fasting blood sugar. This is a study-defined glycemic event rather than a separately verified clinical diagnosis.
_Avoid_: Incident diagnosed diabetes

**At-risk status**:
The study-defined binary glycemic state: positive when cumulative maximum fasting blood sugar exceeds 100 mg/dL, negative when it is at most 100 mg/dL, and unknown when no reading is available through that year. It combines the study's pre-diabetes (>100 through 125 mg/dL) and diabetes (>125 mg/dL) categories.
_Avoid_: Confirmed diabetes diagnosis

**Shared evaluation cohort**:
The intersection of eligible test prediction occasions across the compared models at a given horizon, with the same patients, source years, target years, and target statuses.
_Avoid_: Full source cohort when referring to the comparison population

**Pure-prediction track**:
The study's comparison of future-risk predictions for passive screening, without requiring directional consistency under favorable what-if changes.

**Intervention-safe track**:
The study's monotonic-model track for directionally consistent what-if scores under specified favorable scenarios. Intervention-safe denotes the stated directional criterion, not clinical safety or a causal estimate of intervention benefit.
_Avoid_: Causal treatment effect, guaranteed clinical benefit

**Directional consistency**:
The property that a specified favorable scenario does not increase predicted risk beyond the declared numerical tolerance.

**Calendar-time ablation**:
A comparison that removes calendar-year predictors to assess their contribution while retaining the patient's other predictors.

**PR-AUC in the shared leaderboard**:
Average precision of the at-risk predictions across recall levels, used as the study's primary ranking metric.
_Avoid_: Trapezoidal precision-recall area as an interchangeable definition

**Retrospective benchmark leader**:
The candidate with the highest reported test-set ranking metric among the compared configurations at a given horizon. Its selection and reported performance use the same held-out comparison, so it is not a separately evaluated model-selection procedure.
_Avoid_: Independently validated selected model
