import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

MODEL_NAME = "soccer_xg_model"


def evaluate_and_promote(model, x_test: pd.DataFrame, y_test: pd.Series) -> bool:
    """Evaluates model performance and promotes to 'champion' alias if criteria are met."""
    probs = model.predict_proba(x_test)[:, 1]

    auc = float(roc_auc_score(y_test, probs))
    loss = float(log_loss(y_test, probs))
    brier = float(brier_score_loss(y_test, probs))

    # Log metrics to current active MLflow run
    mlflow.log_metrics({"eval_auc": auc, "eval_log_loss": loss, "eval_brier": brier})
    print(f"\n[Evaluation Metrics] ROC-AUC: {auc:.4f} | Brier Score: {brier:.4f} | Log Loss: {loss:.4f}")

    # Baseline gate: reasonable minimum requirements for soccer xG
    if auc < 0.70 or brier > 0.12:
        print("❌ Model failed minimal quality thresholds. Skipping promotion.")
        return False

    client = MlflowClient()

    # Retrieve current champion metrics if one exists
    current_auc = 0.0
    try:
        current_champ = client.get_model_version_by_alias(MODEL_NAME, "champion")
        run_data = client.get_run(current_champ.run_id).data.metrics
        current_auc = float(run_data.get("eval_auc", 0.0))
        print(f"Current Champion AUC: {current_auc:.4f}")
    except Exception:
        print("No existing champion found. This candidate will be the first champion.")

    # Promotion condition
    if auc >= current_auc:
        latest_versions = client.get_latest_versions(MODEL_NAME)
        if latest_versions:
            new_version = latest_versions[0].version
            client.set_registered_model_alias(MODEL_NAME, "champion", new_version)
            print(f"🏆 Candidate version {new_version} promoted to 'champion' alias!")
            return True

    print("Candidate did not outperform the current champion. Promotion skipped.")
    return False