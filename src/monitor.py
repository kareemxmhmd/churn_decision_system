import json
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp

from src import config
from src.inference import load_inference_artifacts, predict_churn

RAW_NUMERIC_COLS: list[str] = [
    "Tenure",
    "WarehouseToHome",
    "NumberOfDeviceRegistered",
    "SatisfactionScore",
    "NumberOfAddress",
    "DaySinceLastOrder",
    "CashbackAmount",
]

RAW_CATEGORICAL_COLS: list[str] = [
    "PreferedOrderCat",
    "MaritalStatus",
    "Complain",
]


def calculate_psi(
          expected: pd.Series, actual: pd.Series, bin: int = 10
) -> float:

     exp_clean = expected.dropna().to_numpy()
     act_clean = actual.dropna().to_numpy()

     if len(exp_clean) == 0 or len(act_clean) == 0:
          return 0.0

     quantiles = np.linspace(0, 100, bin + 1)
     bin_edges = np.unique(np.percentile(exp_clean, quantiles))
     if len(bin_edges) < 2:
          return 0.0

     bin_edges[0] = -np.inf
     bin_edges[-1] = np.inf

     exp_counts, _ = np.histogram(exp_clean, bins=bin_edges)
     act_counts, _ = np.histogram(act_clean, bins=bin_edges)

     exp_pct = np.clip(exp_counts / len(exp_clean), 1e-4, None)
     act_pct = np.clip(act_counts / len(act_clean), 1e-4, None)

     psi = float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))
     return round(psi, 4)


def detect_feature_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    alpha: float = 0.05,
    psi_threshold: float = 0.20,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []

    for col in RAW_NUMERIC_COLS:
        ref_s = reference_df[col].dropna()
        cur_s = current_df[col].dropna()

        stat, p_val = ks_2samp(ref_s, cur_s)
        psi_val = calculate_psi(ref_s, cur_s)
        drifted = bool((p_val < alpha) and (psi_val >= psi_threshold))

        records.append(
            {
                "feature": col,
                "type": "numeric",
                "test": "ks_2samp",
                "statistic": round(float(stat), 4),
                "p_value": round(float(p_val), 4),
                "psi": psi_val,
                "drift_detected": drifted,
            }
        )

    for col in RAW_CATEGORICAL_COLS:
        ref_counts = reference_df[col].astype(str).value_counts()
        cur_counts = current_df[col].astype(str).value_counts()

        contingency = pd.DataFrame({"ref": ref_counts, "cur": cur_counts}).fillna(0)
        stat, p_val, _, _ = chi2_contingency(contingency.to_numpy())
        drifted = bool(p_val < alpha)

        records.append(
            {
                "feature": col,
                "type": "categorical",
                "test": "chi2",
                "statistic": round(float(stat), 4),
                "p_value": round(float(p_val), 4),
                "psi": None,
                "drift_detected": drifted,
            }
        )

    return pd.DataFrame(records)


def detect_prediction_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    alpha: float = 0.05,
) -> dict[str, Any]:
    
    model, threshold = load_inference_artifacts()

    ref_preds = predict_churn(reference_df, model=model, threshold=threshold)
    cur_preds = predict_churn(current_df, model=model, threshold=threshold)

    ref_proba = ref_preds["churn_probability"]
    cur_proba = cur_preds["churn_probability"]

    stat, p_val = ks_2samp(ref_proba, cur_proba)
    psi_val = calculate_psi(ref_proba, cur_proba)

    return {
        "ks_statistic": round(float(stat), 4),
        "p_value": round(float(p_val), 4),
        "psi": psi_val,
        "ref_positive_rate": round(float(ref_preds["churn_prediction"].mean()), 4),
        "cur_positive_rate": round(float(cur_preds["churn_prediction"].mean()), 4),
        "prediction_drift_detected": bool((p_val < alpha) and (psi_val >= 0.20)),
    }


def main() -> None:
    reference_df = pd.read_csv(config.TEST_DATA)

    current_df = reference_df.sample(frac=0.8, random_state=42).copy()
    current_df["Tenure"] = np.clip(current_df["Tenure"] * 0.4, 0, None)
    current_df["Complain"] = np.random.choice([0, 1], size=len(current_df), p=[0.45, 0.55])

    feature_drift_df = detect_feature_drift(reference_df, current_df)
    pred_drift = detect_prediction_drift(reference_df, current_df)

    drifted_features = feature_drift_df[feature_drift_df["drift_detected"]]["feature"].tolist()
    retrain_needed = len(drifted_features) >= 2 or pred_drift["prediction_drift_detected"]

    print("Feature Drift Summary")
    print(feature_drift_df.to_string(index=False))
    print("\nPrediction Drift Summary")
    print(json.dumps(pred_drift, indent=2))
    print(f"\nDrifted Features: {drifted_features}")
    print(f"Retraining Recommended: {'YES ⚠️' if retrain_needed else 'NO ✅'}")

    report = {
        "retraining_recommended": retrain_needed,
        "drifted_features": drifted_features,
        "prediction_drift": pred_drift,
        "feature_drift": feature_drift_df.to_dict(orient="records"),
    }
    report_path = config.PROCESSED_DIR / "drift_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved drift report to: {report_path}")


if __name__ == "__main__":
    main()