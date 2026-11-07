import numpy as np
import pytest
from fastapi.testclient import TestClient

from mlforge.api.app import create_app
from mlforge.config import InferenceConfig
from mlforge.exceptions import NumericalInstabilityError
from mlforge.inference import Predictor

MAX_BATCH = 5


@pytest.fixture
def client(predictor: Predictor):
    app = create_app(
        predictor=predictor, inference_config=InferenceConfig(device="cpu", max_api_batch_size=MAX_BATCH)
    )
    with TestClient(app) as test_client:
        yield test_client


def test_health(client: TestClient, predictor: Predictor) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["model_loaded"] is True
    assert body["n_features"] == predictor.n_features and body["n_classes"] == predictor.n_classes


def test_predict_single(client: TestClient, predictor: Predictor) -> None:
    response = client.post("/predict", json={"features": [0.0] * predictor.n_features})
    assert response.status_code == 200
    body = response.json()
    assert body["label"] in range(predictor.n_classes)
    assert sum(body["probabilities"]) == pytest.approx(1.0)


def test_predict_accepts_null_for_missing_values(client: TestClient, predictor: Predictor) -> None:
    features = [0.0] * predictor.n_features
    features[1] = None
    assert client.post("/predict", json={"features": features}).status_code == 200


def test_predict_rejects_invalid_payloads(client: TestClient, predictor: Predictor) -> None:
    wrong_length = client.post("/predict", json={"features": [0.0] * (predictor.n_features - 1)})
    assert wrong_length.status_code == 422 and "detail" in wrong_length.json()
    assert client.post("/predict", json={"features": ["abc"] * predictor.n_features}).status_code == 422
    assert client.post("/predict", json={}).status_code == 422
    assert client.post("/predict", json={"features": []}).status_code == 422


def test_predict_batch_matches_predictor(client: TestClient, predictor: Predictor) -> None:
    rows = [[float(i)] * predictor.n_features for i in range(3)]
    response = client.post("/predict/batch", json={"instances": rows})
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 3 and len(body["labels"]) == 3
    expected = predictor.predict_proba(rows)
    np.testing.assert_allclose(body["probabilities"], expected, atol=1e-9)


def test_batch_limits_and_validation(client: TestClient, predictor: Predictor) -> None:
    too_many = [[0.0] * predictor.n_features] * (MAX_BATCH + 1)
    assert client.post("/predict/batch", json={"instances": too_many}).status_code == 413
    assert client.post("/predict/batch", json={"instances": []}).status_code == 422
    ragged = [[0.0] * predictor.n_features, [0.0] * (predictor.n_features - 1)]
    assert client.post("/predict/batch", json={"instances": ragged}).status_code == 422


def test_internal_library_errors_return_500(client: TestClient, predictor: Predictor, monkeypatch) -> None:
    def broken(*args, **kwargs):
        raise NumericalInstabilityError("simulated NaN output")

    monkeypatch.setattr(predictor, "predict_proba", broken)
    response = client.post("/predict", json={"features": [0.0] * predictor.n_features})
    assert response.status_code == 500 and "simulated NaN output" in response.json()["detail"]


def test_missing_checkpoint_degrades_gracefully(tmp_path, predictor: Predictor) -> None:
    app = create_app(checkpoint_path=tmp_path / "missing.pt", inference_config=InferenceConfig(device="cpu"))
    with TestClient(app) as test_client:
        health = test_client.get("/health").json()
        assert health["status"] == "degraded" and health["model_loaded"] is False
        response = test_client.post("/predict", json={"features": [0.0] * predictor.n_features})
        assert response.status_code == 503


def test_loads_model_from_checkpoint_at_startup(trained_run) -> None:
    app = create_app(
        checkpoint_path=trained_run.checkpoint, inference_config=InferenceConfig(device="cpu")
    )
    with TestClient(app) as test_client:
        assert test_client.get("/health").json()["model_loaded"] is True