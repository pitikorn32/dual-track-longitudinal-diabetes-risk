import json

import pytest


@pytest.mark.parametrize("field", ["FBS", "BMI", "Pulse", "BL_pres1", "BL_pres2", "Waist",
                                  "total_sugary_week", "total_veg_fruit_week", "total_exercise_week",
                                  "total_phy_activity_week", "sleep_hours", "max_fbs_to_date",
                                  "years_since_last_fbs"])
def test_request_rejects_nonfinite_measurements_and_questionnaire(api, field):
    for value in [float("inf"), float("-inf"), float("nan")]:
        payload = {"horizon_years": 1, "history_years": 1, "age": 45, "measurements": [{}]}
        target = payload["measurements"][0] if field in api.CLINICAL_FEATURES else payload
        target[field] = value
        with pytest.raises(ValueError):
            api.PredictRequest(**payload)


@pytest.mark.parametrize("route", ["/predict", "/predict/interventions", "/no_year/predict",
                                  "/no_year/predict/interventions", "/logistic_only/predict",
                                  "/logistic_only/predict/interventions", "/logistic_only/no_year/predict",
                                  "/logistic_only/no_year/predict/interventions"])
@pytest.mark.parametrize("value", ["1e309", "-1e309", "NaN"])
def test_invalid_number_returns_json_422_before_scoring(api, request_api, monkeypatch, route, value):
    def unexpected_scoring(*args, **kwargs):
        pytest.fail("Invalid numbers must be rejected before selecting or scoring a model")

    monkeypatch.setattr(api, "_get_artifact", unexpected_scoring)
    body = ('{"horizon_years":1,"history_years":1,"age":45,"presets":["reduce_bmi_by_one"],'
            '"measurements":[{"FBS":' + value + '}]}')
    status, result = request_api(api.app, route, body=body)
    assert status == 422
    error = result["detail"][0]
    assert error["loc"] == ["body", "measurements", 0, "FBS"]
    assert error["type"] == "finite_number"
    assert "input" not in error


def test_missing_measurements_keep_existing_null_semantics(api):
    req = api.PredictRequest(horizon_years=1, history_years=1, age=45, measurements=[{}])
    assert req.measurements[0].FBS is None
    assert req.total_exercise_week is None


def test_measurement_length_error_remains_json_serializable(api, request_api):
    status, result = request_api(api.app, "/predict", body=json.dumps({
        "horizon_years": 1, "history_years": 3, "age": 45, "measurements": [{}],
    }))
    assert status == 422
    assert "exactly history_years=3" in result["detail"][0]["msg"]
