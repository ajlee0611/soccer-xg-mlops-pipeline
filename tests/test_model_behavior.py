import numpy as np
import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

from serving.app import app, model_store

client = TestClient(app)

BASE_PAYLOAD = {
    "match_id": 3869685,
    "minute": 34,
    "player_name": "Test Striker",
    "play_pattern": "Regular Play",
    "shot_type": "Open Play",
    "body_part": "Right Foot",
    "under_pressure": False,
    "first_time": False,
}


@pytest.fixture(autouse=True)
def mock_serving_model():
    """Inject a mock model directly into model_store handling raw numpy arrays."""
    mock_model = MagicMock()

    def mock_predict_proba(features):
        arr = np.asarray(features)
        # Ensure 2D shape (n_samples, n_features)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        results = []
        for row in arr:
            # First feature in soccer xG vectors is distance (or dx/dy)
            val = float(row[0])
            # If val is distance in yards, 4 yards (tap-in) yields ~0.62, 45 yards yields ~0.12
            prob = max(0.01, min(0.95, 1.0 / (1.0 + max(0.1, val) * 0.15)))
            results.append([1.0 - prob, prob])

        return np.array(results)

    def mock_predict(features):
        probs = mock_predict_proba(features)
        return probs[:, 1]

    mock_model.predict_proba = mock_predict_proba
    mock_model.predict = mock_predict

    previous_model = model_store.get("model")
    model_store["model"] = mock_model
    yield
    if previous_model is not None:
        model_store["model"] = previous_model
    else:
        model_store.pop("model", None)


def test_prediction_output_bounds():
    """Verify xG probabilities strictly adhere to [0.0, 1.0]."""
    payload = {
        **BASE_PAYLOAD,
        "location_x": 105.0,
        "location_y": 40.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    xg = response.json()["xg"]
    assert 0.0 <= xg <= 1.0


def test_distance_monotonicity():
    """ML Invariant: A tap-in must have higher xG than a midfield strike."""
    close_shot = {
        **BASE_PAYLOAD,
        "location_x": 116.0,
        "location_y": 40.0,
    }
    long_range_shot = {
        **BASE_PAYLOAD,
        "location_x": 75.0,
        "location_y": 40.0,
    }

    close_res = client.post("/predict", json=close_shot).json()
    long_res = client.post("/predict", json=long_range_shot).json()

    assert "xg" in close_res, f"Key 'xg' missing from response: {close_res}"
    assert "xg" in long_res, f"Key 'xg' missing from response: {long_res}"
    assert close_res["xg"] > long_res["xg"], (
        f"Model failed directional sanity check: close ({close_res['xg']}) "
        f"<= long ({long_res['xg']})"
    )