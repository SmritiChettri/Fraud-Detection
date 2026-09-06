import pytest
from fastapi.testclient import TestClient
from app.main import app

@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client

def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict_endpoint(client):
    payload = {
        "Time": 100.0,
        "Amount": 45.50,
        **{f"V{i}": 0.05 for i in range(1, 29)}
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200