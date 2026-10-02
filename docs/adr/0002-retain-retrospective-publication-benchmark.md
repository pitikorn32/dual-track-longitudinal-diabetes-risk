---
status: accepted
---

# Retain the retrospective publication benchmark

The HealthCom analysis compares candidate models on shared held-out patient-year rows and identifies the nominal leaders from their test-set metrics. Retain that retrospective comparison in the publication workflow, explicitly distinguishing it from an independently evaluated model-selection procedure. Introducing validation-based selection during this refactor would change the experiment described by the fixed paper.

Validation-based selection for deployment remains separate future work. Correcting implementation errors, using consistent metric definitions, attaching directional checks to the exact evaluated model configuration, and enforcing shared evaluation rows do not authorize a new model-selection protocol.
