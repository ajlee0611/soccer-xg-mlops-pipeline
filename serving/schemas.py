from pydantic import BaseModel, ConfigDict, Field

class ShotRequest(BaseModel):
    match_id: int
    minute: int = Field(..., ge=0, le=130)
    player_name: str
    location_x: float = Field(..., ge=0.0, le=120.0)
    location_y: float = Field(..., ge=0.0, le=80.0)
    body_part: str
    play_pattern: str
    under_pressure: bool = False
    first_time: bool = False

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "match_id": 3869685,
                "minute": 67,
                "player_name": "Lionel Messi",
                "location_x": 104.5,
                "location_y": 39.0,
                "body_part": "Left Foot",
                "play_pattern": "Regular Play",
                "under_pressure": False,
                "first_time": False,
            }
        }
    )


class PredictionResponse(BaseModel):
    match_id: int
    player_name: str
    xg: float
    distance_yards: float
    distance_meters: float
    angle_degrees: float