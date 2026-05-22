"""Pydantic request/response schemas for the DigiHealth Risk Score API.

Extracted from api.py so the wire contract lives in one place. The same request
body is accepted by every /predict* endpoint; the intervention endpoints add a
non-empty `presets` list.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Input schemas
# ---------------------------------------------------------------------------

class ClinicalMeasurement(BaseModel):
    """One annual health checkup. All clinical fields are optional (null = not measured that year)."""
    FBS: float | None = Field(None, description="Fasting blood sugar (mg/dL).")
    BMI: float | None = Field(None, description="Body mass index (kg/m²).")
    Pulse: float | None = Field(None, description="Pulse rate (bpm).")
    BL_pres1: float | None = Field(None, description="Systolic blood pressure (mmHg).")
    BL_pres2: float | None = Field(None, description="Diastolic blood pressure (mmHg).")
    Waist: float | None = Field(None, description="Waist circumference (cm).")


class PredictRequest(BaseModel):
    """
    Anonymous risk prediction request.

    `measurements` must contain exactly `history_years` entries ordered
    oldest → newest. The last entry is the current (most recent) checkup.
    Missing checkup years are represented by null clinical values.
    """
    horizon_years: Literal[1, 2, 3, 4, 5] = Field(
        ..., description="Predict risk N years ahead of the most recent measurement."
    )
    history_years: Literal[1, 3, 5] = Field(
        ..., description="Number of annual measurements provided (must match len(measurements))."
    )

    # Demographics
    age: int = Field(..., ge=1, le=120, description="Patient age at time of most recent measurement.")
    year: int | None = Field(
        None,
        description=(
            "Calendar year of the most recent measurement (e.g. 2024). "
            "Used for the Year_centered temporal feature. "
            "Defaults to the current server year if omitted."
        ),
    )

    # Questionnaire (all optional — model handles nulls via median imputation)
    gender: str | None = None
    dm_first_degree_relative: bool | None = None
    cooking_method: str | None = None
    total_sugary_week: float | None = None
    total_veg_fruit_week: float | None = None
    total_exercise_week: float | None = None
    total_phy_activity_week: float | None = None
    sleep_hours: float | None = None
    sleep_quality: str | None = None
    smoking_status: str | None = None
    alcohol_status: str | None = None

    # Optional cumulative history aggregates. When omitted, the API derives each
    # from `measurements` alone, which only matches training-time semantics if
    # the submitted window covers the patient's full prior FBS history. Supply
    # them explicitly for short windows (especially M=1) to preserve the Phase 0
    # `MAX_FBS_up_to_year`, `years_since_last_fbs`, and `is_missing_last_year`
    # semantics that the training tables were built with.
    max_fbs_to_date: float | None = Field(
        None,
        description=(
            "Cumulative maximum FBS observed from baseline through the most "
            "recent measurement (mg/dL). Defaults to max(submitted FBS) when "
            "omitted; supply explicitly when the submitted window does not "
            "cover the full prior FBS history."
        ),
    )
    years_since_last_fbs: float | None = Field(
        None,
        ge=0,
        description=(
            "Years between the most recent prior observed FBS and the most "
            "recent measurement (0 if the current measurement has FBS). "
            "Defaults to the gap within `measurements` when omitted; supply "
            "explicitly to preserve training-time semantics for short windows."
        ),
    )
    previous_year_fbs_missing: bool | None = Field(
        None,
        description=(
            "True if no FBS was observed in the calendar year immediately "
            "before `year`. Defaults to `measurements[-2].FBS is None` when "
            "len(measurements) >= 2, otherwise null. Supply explicitly at M=1 "
            "to match training-time `is_missing_last_year`."
        ),
    )

    # Clinical measurements: exactly history_years entries, oldest first
    measurements: list[ClinicalMeasurement] = Field(
        ...,
        description="Annual checkup records ordered oldest → newest. Length must equal history_years.",
    )

    @model_validator(mode="after")
    def check_measurements_length(self) -> "PredictRequest":
        if len(self.measurements) != self.history_years:
            raise ValueError(
                f"measurements must have exactly history_years={self.history_years} entries, "
                f"got {len(self.measurements)}."
            )
        return self

    model_config = {"json_schema_extra": {
        "example": {
            "horizon_years": 3,
            "history_years": 5,
            "age": 47,
            "year": 2024,
            "gender": "female",
            "dm_first_degree_relative": False,
            "total_sugary_week": 3.0,
            "total_veg_fruit_week": 4.0,
            "total_exercise_week": 2.0,
            "total_phy_activity_week": 3.0,
            "max_fbs_to_date": 110.0,
            "years_since_last_fbs": 0.0,
            "previous_year_fbs_missing": False,
            "measurements": [
                {"FBS": 90.0,  "BMI": 25.5, "Pulse": 70.0, "BL_pres1": 120.0, "BL_pres2": 78.0, "Waist": 82.0},
                {"FBS": 95.0,  "BMI": 26.0, "Pulse": 72.0, "BL_pres1": 122.0, "BL_pres2": 79.0, "Waist": 83.0},
                {"FBS": None,  "BMI": 26.2, "Pulse": None,  "BL_pres1": None,  "BL_pres2": None,  "Waist": 84.0},
                {"FBS": 100.0, "BMI": 26.5, "Pulse": 74.0, "BL_pres1": 128.0, "BL_pres2": 82.0, "Waist": 86.0},
                {"FBS": 105.0, "BMI": 27.0, "Pulse": 76.0, "BL_pres1": 130.0, "BL_pres2": 85.0, "Waist": 88.0},
            ],
        }
    }}


class InterventionRequest(PredictRequest):
    """Same as PredictRequest but also specifies which intervention presets to evaluate."""
    presets: list[str] = Field(
        ..., min_length=1,
        description="Named intervention presets to evaluate against the baseline.",
    )


# ---------------------------------------------------------------------------
# Output schemas
# ---------------------------------------------------------------------------

class RiskResult(BaseModel):
    probability: float
    risk_score: float = Field(..., description="Risk score 0–100 (probability × 100).")
    threshold: float
    at_risk_flag: bool = Field(..., description="True if probability >= threshold.")


class PredictResponse(RiskResult):
    model_key: str
    track: str
    model_family: str
    horizon_years: int
    history_years: int

    model_config = {"protected_namespaces": ()}


class ScenarioResult(BaseModel):
    preset: str
    description: str
    probability: float
    risk_score: float
    delta_risk_score: float = Field(..., description="Scenario score minus baseline score.")
    at_risk_flag: bool
    changed_features: dict[str, dict[str, float | None]]


class InterventionResponse(BaseModel):
    model_key: str
    track: str
    model_family: str
    horizon_years: int
    history_years: int
    baseline: RiskResult
    scenarios: list[ScenarioResult]

    model_config = {"protected_namespaces": ()}


class ModelInfo(BaseModel):
    key: str
    track: str
    model_family: str
    horizon_years: int
    history_years: int
    threshold: float
    feature_count: int
    intervention_presets: list[str]

    model_config = {"protected_namespaces": ()}
