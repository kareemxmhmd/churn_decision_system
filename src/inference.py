import json
from typing import Union

import joblib
import pandas as pd
from sklearn.base import BaseEstimator

from src import config


def load_inference_artifacts() -> tuple[BaseEstimator, float]:
    if not config.MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at {config.MODEL_PATH}. Run 'python -m src.train' first."
        )

    model: BaseEstimator = joblib.load(config.MODEL_PATH)

    threshold = 0.5
    if config.THRESHOLD_PATH.exists():
        threshold_data = json.loads(config.THRESHOLD_PATH.read_text(encoding="utf-8"))
        threshold = float(threshold_data.get("threshold", 0.5))

    return model, threshold


def predict_churn(
    data: Union[pd.DataFrame, dict, list[dict]],
    model: BaseEstimator | None = None,
    threshold: float | None = None,
) -> pd.DataFrame:

    if model is None or threshold is None:
        loaded_model, loaded_threshold = load_inference_artifacts()
        model = model or loaded_model
        threshold = threshold if threshold is not None else loaded_threshold

    if isinstance(data, dict):
        df = pd.DataFrame([data])
    elif isinstance(data, list):
        df = pd.DataFrame(data)
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise TypeError("Input data must be a pandas DataFrame, dict, or list of dicts.")

    features = df.drop(columns=[config.TARGET], errors="ignore")

    proba = model.predict_proba(features)[:, 1]
    preds = (proba >= threshold).astype(int)

    results = pd.DataFrame(
        {
            "churn_probability": proba.round(4),
            "churn_prediction": preds,
            "threshold_used": round(threshold, 4),
        },
        index=df.index,
    )
    return results


def main() -> None:
    model, threshold = load_inference_artifacts()
    print(f"Loaded model from: {config.MODEL_PATH}")
    print(f"Using optimal threshold: {threshold:.4f}\n")

    sample_df = pd.read_csv(config.TEST_DATA).head(5)
    predictions = predict_churn(sample_df, model=model, threshold=threshold)

    if config.TARGET in sample_df.columns:
        predictions["actual_churn"] = sample_df[config.TARGET].to_numpy()

    print("Sample Predictions:")
    print(predictions.to_string(index=True))


if __name__ == "__main__":
    main()