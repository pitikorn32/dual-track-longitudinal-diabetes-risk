---
status: accepted
---

# Prioritize HealthCom methods with supplementary thesis analyses

The shared research repository supports the fixed HealthCom camera-ready paper as its primary publication and the extended thesis as supplementary analyses. The HealthCom manuscript cannot be edited. The release should cover the union of their reported analyses, with HealthCom reproduction commands and documentation presented first and thesis extensions separately identified.

The private source cohort will not be published. The public release explains and implements the methods, maps publication claims to runnable analyses, and provides synthetic smoke checks for readers without the cohort. It does not promise that those readers can reproduce the private-cohort numbers. Exact numerical agreement with historical results is not a release requirement: modest differences are acceptable when they do not materially contradict the paper, while discrepancies and their provenance must remain explicit.

Preserve available historical evidence before changing scientific behavior, and identify corrected analyses separately from the published results. Establish and validate the research workflow before changing deployment behavior. Existing deployment fits remain clearly identified as separate variants; serving the exact benchmark models is a separate acceptance decision. This scope does not merge the downstream BHI exporter into the HealthCom benchmark.

Fix verified implementation problems even when their correction changes outputs. For discrepancies between manuscript descriptions and implementation, establish the actual behavior and document it in repository notes; the camera-ready manuscript remains unchanged. Close changes in winner order are acceptable, while changes to central conclusions, including directional consistency and the comparative performance of the two tracks, require investigation before release. Preserving the paper's direction is a preference to assess against results, not a reason to alter or conceal them.

The starting code revision for this refactor is `0ca3c54f6a6ae41f8fa93f58fd004c043dcc282d`. It is a recoverable code reference, not a verified identification of the revision that produced every published number. Existing local artifacts must be preserved separately before reruns overwrite them.

## Consequences

Public usability and faithful methods take priority over forcing a rerun to match every printed decimal. Material differences must be reported with their causes or remaining uncertainty. Neither an absence of public patient data nor tolerance for small numerical differences removes the need to check endpoint construction, patient separation, temporal leakage, shared evaluation rows, or directional-safety evidence.

## Confirmed implementation sequence

1. Preserve existing local result artifacts before reruns and record the starting code revision, configurations, and installed `digihealth` dependency versions. Keep private inputs and patient-level outputs outside tracked release content.
2. Map HealthCom claims, numerical tables, scientific figures, bootstrap inference, and ablations to commands and expected output schemas. Provide a supplementary mapping for thesis extensions. Distinguish published reference values from newly generated results.
3. Consolidate shared research logic where duplication can cause disagreements: endpoint validation, canonical patient splitting, average-precision and ROC-AUC evaluation, feature transformations, and exact patient-year alignment. Keep familiar phase entry points usable and preserve downstream deployment and BHI interfaces.
4. Correct verified orchestration and evaluation defects: argument parsing, exact model/configuration matching for safety results, consistent directional-check scenarios and tolerance units, all-row safety evaluation, and recomputation of scenario-dependent derived features. Retain the retrospective benchmark and documented family-specific training/calibration allocations.
5. Bring the HealthCom bootstrap analysis into the shared research package with portable paths and explicit shared-cohort alignment. Fill gaps in the publication command mapping and make HealthCom and supplementary thesis workflows separately runnable.
6. Add meaningful tests for thresholds and unknown labels, patient separation, future clinical-feature exclusion, metric consistency, shared-cohort identity, and scenario-score direction. Provide a deterministic, wholly synthetic example and smoke workflow that require no private cohort.
7. Run relevant checks and affected analyses using `digihealth`, broadening to the agreed publication workflows as needed. Compare corrected outputs with preserved references and assess changes to the paper's central conclusions. Record executable commands and dependency versions; do not silently replace the user's environment to chase reference numbers.
8. Present HealthCom first in the repository README, with thesis supplements, data schema, commands, method clarifications, static-lifestyle limitations, and separately identified deployment variants. Deliver the local changes and validation report for review; publishing or pushing a release is outside this design confirmation.

Acceptance requires executable documented workflows, passing scientific-invariant checks, and an explanation of material changes. It does not require identical printed decimals, verified questionnaire collection dates that are unavailable, or new prospective model-selection experiments. The author confirmed this sequence and requested separate commits for independently reviewable and reversible steps.
