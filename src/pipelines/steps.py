from pathlib import Path
from typing import Tuple
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from zenml import step

from src.data.ingest import extract_shots_dataset
from src.models.evaluate import evaluate_and_promote

FEATURE_COLUMNS = [
    "distance_to_goal",
    "shot_angle_rad",
    "is_header",
    "is_open_play",
    "under_pressure",
    "first_time",
]


@step
def ingest_data_step() -> pd.DataFrame:
    """Ingests raw StatsBomb data or loads cached local parquet file."""
    parquet_path = Path("data/raw_shots.parquet")
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    return extract_shots_dataset()


@step
def prepare_features_step(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Splits data into train and test sets with stratification."""
    x = df[FEATURE_COLUMNS]
    y = df["target"]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.20, random_state=42, stratify=y
    )
    return x_train, x_test, y_train, y_test


@step(enable_cache=False)
def train_xg_step(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 120,
    max_depth: int = 3,
    learning_rate: float = 0.05,
) -> XGBClassifier:
    """Trains an XGBoost classifier with autologged MLflow artifacts."""
    with mlflow.start_run(nested=True):
        mlflow.log_params(
            {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "learning_rate": learning_rate,
            }
        )

        model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
        )
        model.fit(x_train, y_train)

        # Register the model into MLflow Model Registry
        mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path="model",
            registered_model_name="soccer_xg_model",
        )
        return model


@step(enable_cache=False)
def evaluate_model_step(
    model: XGBClassifier,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> bool:
    """Runs evaluation gate against holdout dataset."""
    return evaluate_and_promote(model, x_test, y_test)