from zenml import pipeline
from src.pipelines.steps import (
    ingest_data_step,
    prepare_features_step,
    train_xg_step,
    evaluate_model_step,
)


@pipeline(enable_cache=False)
def soccer_xg_training_pipeline():
    """End-to-end training and promotion pipeline."""
    df = ingest_data_step()
    x_train, x_test, y_train, y_test = prepare_features_step(df)
    model = train_xg_step(x_train, y_train)
    evaluate_model_step(model, x_test, y_test)


if __name__ == "__main__":
    print("Executing ZenML Soccer xG Pipeline...")
    soccer_xg_training_pipeline()
    print("\nPipeline execution complete!")