from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parents[1]

# Data Paths
RAW_DATA = ROOT / "data" / "raw" / "customer_churn.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
OOF_TRAIN_PATH = PROCESSED_DIR / "oof_train.csv"
TEST_DATA = PROCESSED_DIR / "test_set.csv"
METRICS_PATH = PROCESSED_DIR / "metrics.json"
PLOTS_DIR = PROCESSED_DIR / "plots"

# Model Paths
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "churn_pipeline.pkl"
THRESHOLD_PATH = MODELS_DIR / "threshold.json"

# Target and Experimentation
TARGET = "Churn"
F_BETA = 2.0  # Beta > 1 weights recall higher than precision during threshold tuning

# Cost Constants (Business Cost Matrix in USD)
COST_FALSE_NEGATIVE = 500  # Cost of missed churning customer (lost customer LTV)
COST_FALSE_POSITIVE = 50   # Cost of unnecessary retention incentive/discount