"""FastAPI service serving the production Champion xG model."""

from contextlib import asynccontextmanager
from pathlib import Path
import mlflow.xgboost
import numpy as np
from xgboost import XGBClassifier
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from serving.schemas import ShotRequest, PredictionResponse

model_store = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loads the model at startup (tries MLflow registry first, then local artifact)."""
    loaded = False

    # 1. Try MLflow Registry
    try:
        model_uri = "models:/soccer_xg_model@champion"
        model_store["model"] = mlflow.xgboost.load_model(model_uri)
        print("✅ Production Champion xG model loaded from MLflow.")
        loaded = True
    except Exception as e:
        print(f"ℹ️ MLflow registry load skipped/unavailable: {e}")

    # 2. Fallback to packaged model file
    if not loaded:
        local_path = Path("models/champion_model.json")
        if local_path.exists():
            model = XGBClassifier()
            model.load_model(str(local_path))
            model_store["model"] = model
            print("✅ Production Champion xG model loaded from local artifact.")
            loaded = True
        else:
            print("❌ No local model artifact found at models/champion_model.json.")

    yield
    model_store.clear()


app = FastAPI(
    title="Soccer Expected Goals (xG) Microservice",
    description="Real-time probability inference engine using StatsBomb pitch telemetry.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for browser frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": "model" in model_store,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_xg(shot: ShotRequest):
    if "model" not in model_store:
        raise HTTPException(
            status_code=503,
            detail="Model is currently unavailable or uninitialized.",
        )

    # 1. Spatial Math: Distance to goal center (120, 40)
    dx = 120.0 - shot.location_x
    dy = 40.0 - shot.location_y
    dist_yards = float(np.sqrt(dx**2 + dy**2))
    dist_meters = float(dist_yards * 0.9144)

    # 2. Spatial Math: Angle between goal posts (120, 36) and (120, 44)
    v1_x = 120.0 - shot.location_x
    v1_y = 36.0 - shot.location_y
    v2_x = 120.0 - shot.location_x
    v2_y = 44.0 - shot.location_y

    dot = v1_x * v2_x + v1_y * v2_y
    mag1 = np.sqrt(v1_x**2 + v1_y**2)
    mag2 = np.sqrt(v2_x**2 + v2_y**2)

    cosine = np.clip(dot / (mag1 * mag2), -1.0, 1.0)
    angle_rad = float(np.arccos(cosine))
    angle_deg = float(np.degrees(angle_rad))

    # 3. Format features for the trained XGBoost model
    features = np.array(
        [
            [
                dist_yards,
                angle_rad,
                1 if shot.body_part == "Head" else 0,
                1 if shot.play_pattern == "Regular Play" else 0,
                1 if shot.under_pressure else 0,
                1 if shot.first_time else 0,
            ]
        ]
    )

    # 4. Predict goal probability
    model = model_store["model"]
    xg_probability = float(model.predict_proba(features)[0][1])

    return PredictionResponse(
        match_id=shot.match_id,
        player_name=shot.player_name,
        xg=round(xg_probability, 4),
        distance_yards=round(dist_yards, 2),
        distance_meters=round(dist_meters, 2),
        angle_degrees=round(angle_deg, 2),
    )