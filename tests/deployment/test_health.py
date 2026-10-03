import pytest


VARIANTS = [
    ("/health", "_models", "_expected_keys", "_load_all_models", "MODEL_DIR"),
    ("/no_year/health", "_models_no_year", "_expected_keys", "_load_all_models_no_year", "MODEL_DIR_NO_YEAR"),
    ("/logistic_only/health", "_models_logistic_only", "_expected_keys_logistic_only", "_load_all_models_logistic_only", "MODEL_DIR_LOGISTIC_ONLY"),
    ("/logistic_only/no_year/health", "_models_logistic_only_no_year", "_expected_keys_logistic_only", "_load_all_models_logistic_only_no_year", "MODEL_DIR_LOGISTIC_ONLY_NO_YEAR"),
]


@pytest.mark.parametrize("route,store_name,keys_name,loader_name,directory_name", VARIANTS)
@pytest.mark.parametrize("state", ["empty", "partial", "wrong_key", "complete"])
def test_health_requires_every_expected_model(api, request_api, monkeypatch, route, store_name,
                                            keys_name, loader_name, directory_name, state):
    keys = getattr(api, keys_name)()
    store = {} if state == "empty" else {key: {} for key in keys}
    if state in {"partial", "wrong_key"}:
        del store[keys[-1]]
    if state == "wrong_key":
        store["unrelated-model"] = {}
    monkeypatch.setattr(api, store_name, store)
    status, result = request_api(api.app, route)
    assert status == (200 if state == "complete" else 503)
    assert result["status"] == ("ok" if state == "complete" else
                                "models_not_loaded" if state == "empty" else "models_incomplete")
    assert result["expected"] == 30
    assert result["models_loaded"] == len(store)
    assert result["missing_model_keys"] == [key for key in keys if key not in store]


@pytest.mark.parametrize("route,store_name,keys_name,loader_name,directory_name", VARIANTS)
def test_reloading_does_not_leave_removed_models_ready(api, request_api, monkeypatch, tmp_path, route,
                                                    store_name, keys_name, loader_name, directory_name):
    keys = getattr(api, keys_name)()
    monkeypatch.setattr(api, store_name, {key: {} for key in keys})
    monkeypatch.setattr(api, directory_name, tmp_path)
    getattr(api, loader_name)()
    status, result = request_api(api.app, route)
    assert status == 503
    assert result["models_loaded"] == 0


