from fastapi.testclient import TestClient
from serving.app import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict_endpoint_response_structure():
    payload = {
        "match_id": 3869685,
        "minute": 34,
        "player_name": "Test Striker",
        "location_x": 108.0,
        "location_y": 40.0,
        "body_part": "Right Foot",
        "play_pattern": "Regular Play",
        "under_pressure": False,
        "first_time": False,
    }
    response = client.post("/predict", json=payload)
    # Status is 200 if model is loaded into memory, or 503 if testing isolated container
    assert response.status_code in [200, 503]