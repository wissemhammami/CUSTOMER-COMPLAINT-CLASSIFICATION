from fastapi.testclient import TestClient

from src.api.main import app


def test_predict_returns_label_for_valid_request():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={"text": "My mortgage payment was applied to the wrong account."},
        )

    assert response.status_code == 200
    assert response.json()["predicted_label"] == "Mortgage"


def test_predict_rejects_empty_and_malformed_requests():
    with TestClient(app) as client:
        empty_response = client.post("/predict", json={"text": "   "})
        malformed_response = client.post("/predict", json={"text": 123})
        missing_response = client.post("/predict", json={})

    assert empty_response.status_code == 422
    assert malformed_response.status_code == 422
    assert missing_response.status_code == 422
