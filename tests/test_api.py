"""Pytest suite for MP141 FastAPI prediction service."""

from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from app.main import app


class _DummyModel:
    """Fallback estimator used only when models/model.joblib is not yet present."""

    def predict(self, df):
        return [8500.0 for _ in range(len(df))]


def _valid_payload() -> dict:
    return {
        "distance_to_campus_km": 3.5,
        "room_area_sqft": 180,
        "sharing": 2,
        "meals_per_day": 2,
        "building_age_yr": 25,
        "wifi_mbps": 100,
        "furnishing": "semi",
        "locality": "Sector East",
    }


@pytest.fixture()
def client():
    """Provide a TestClient; inject fallback model if models/model.joblib is absent."""
    had_model_attr = hasattr(app.state, "model")
    original_model = getattr(app.state, "model", None)
    model_exists = Path("models/model.joblib").exists()

    if not model_exists:
        app.state.model = _DummyModel()

    with TestClient(app) as test_client:
        yield test_client

    if not model_exists:
        if had_model_attr:
            app.state.model = original_model
        elif hasattr(app.state, "model"):
            delattr(app.state, "model")


def test_health_endpoint(client: TestClient):
    """GET /health returns HTTP 200 with {'status': 'ok'}."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_valid_input(client: TestClient):
    """POST /predict accepts valid input and returns a numeric prediction."""
    response = client.post("/predict", json=_valid_payload())
    assert response.status_code == 200
    body = response.json()
    assert "prediction" in body
    assert isinstance(body["prediction"], (int, float))


def test_predict_boundary_values(client: TestClient):
    """POST /predict accepts exact boundary values building_age_yr=75 and wifi_mbps=0."""
    payload = _valid_payload()
    payload["building_age_yr"] = 75
    payload["wifi_mbps"] = 0

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "prediction" in response.json()


def test_predict_rejects_building_age_above_75(client: TestClient):
    """POST /predict rejects building_age_yr > 75 with HTTP 422."""
    payload = _valid_payload()
    payload["building_age_yr"] = 76

    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_rejects_negative_wifi(client: TestClient):
    """POST /predict rejects wifi_mbps < 0 with HTTP 422."""
    payload = _valid_payload()
    payload["wifi_mbps"] = -1

    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_rejects_extra_fields(client: TestClient):
    """POST /predict rejects unexpected fields outside the 8 required features."""
    payload = _valid_payload()
    payload["rent_rs"] = 9000

    response = client.post("/predict", json=payload)
    assert response.status_code == 422
