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
