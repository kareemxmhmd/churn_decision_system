from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parents[1]

# Data Paths
RAW_DATA = ROOT / 'data' / 'raw' / 'customer_churn.csv'
# Processed Data Paths
PROCESSED_DIR = ROOT / 'data' / 'processed'
# Out-of-Fold Training Data Path
OOF_TRAIN_PATH = PROCESSED_DIR / "oof_train.csv"
# Test Data Path
TEST_DATA = PROCESSED_DIR / "test_set.csv"
# Metrics Path
METRICS_PATH = PROCESSED_DIR / "metrics.json"
# Model Paths
MODELS_DIR = ROOT / 'models'
MODEL_PATH = MODELS_DIR / 'churn_pipeline.pkl'

# Target and Experimentation
TARGET = 'Churn'

# Cost Constants (Business Cost Matrix in USD)
COST_FALSE_NEGATIVE = 500  # Cost of missed churning customer (lost customer LTV)
COST_FALSE_POSITIVE = 50   # Cost of unnecessary retention incentive/discount