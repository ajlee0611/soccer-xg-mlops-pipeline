import numpy as np
import pandas as pd


def compute_shot_geometry(df: pd.DataFrame) -> pd.DataFrame:
    """Computes Euclidean distance and goal-mouth visible angle in radians.

    Expects df to contain 'location_x' and 'location_y' columns.
    """
    data = df.copy()

    goal_center_x, goal_center_y = 120.0, 40.0
    post1_y, post2_y = 36.0, 44.0

    # 1. Distance to center of the goal
    dx = goal_center_x - data["location_x"]
    dy = goal_center_y - data["location_y"]
    data["distance_to_goal"] = np.sqrt(dx**2 + dy**2)

    # 2. Angle subtended by the goal posts
    v1_x = 120.0 - data["location_x"]
    v1_y = post1_y - data["location_y"]

    v2_x = 120.0 - data["location_x"]
    v2_y = post2_y - data["location_y"]

    dot_product = v1_x * v2_x + v1_y * v2_y
    mag1 = np.sqrt(v1_x**2 + v1_y**2)
    mag2 = np.sqrt(v2_x**2 + v2_y**2)

    # Clip to avoid floating-point overflow outside [-1.0, 1.0]
    cosine_angle = np.clip(dot_product / (mag1 * mag2), -1.0, 1.0)
    data["shot_angle_rad"] = np.arccos(cosine_angle)

    # 3. Binary feature indicators
    data["is_header"] = (data["body_part"] == "Head").astype(int)
    data["is_open_play"] = (data["play_pattern"] == "Regular Play").astype(int)
    data["under_pressure"] = data["under_pressure"].fillna(False).astype(int)
    data["first_time"] = data["first_time"].fillna(False).astype(int)

    return data