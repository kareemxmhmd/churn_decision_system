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