import numpy as np
import pandas as pd
from src.features.build_features import compute_shot_geometry


def test_penalty_spot_geometry():
    """Penalty spot is located at (108, 40) - 12 yards out directly in front of the goal."""
    sample_df = pd.DataFrame(
        [
            {
                "location_x": 108.0,
                "location_y": 40.0,
                "body_part": "Right Foot",
                "play_pattern": "Other",
                "under_pressure": False,
                "first_time": False,
            }
        ]
    )

    result = compute_shot_geometry(sample_df)

    # Distance must be exactly 12 yards
    assert np.isclose(result["distance_to_goal"].iloc[0], 12.0, atol=1e-2)

    # Visible angle from penalty spot should be roughly 0.638 radians (~36.5 degrees)
    expected_angle = 2 * np.arctan(4 / 12)
    assert np.isclose(result["shot_angle_rad"].iloc[0], expected_angle, atol=1e-2)