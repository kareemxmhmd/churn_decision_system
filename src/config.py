from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = ROOT / 'data' / 'raw' / 'customer_churn.csv'
PROCESSED_DATA_DIR = ROOT / 'data' / 'processed'
TRAIN_DATA = PROCESSED_DATA_DIR / 'train.csv'
TEST_DATA = PROCESSED_DATA_DIR / 'test.csv'

MODELS_DIR = ROOT / 'models'
MODEL_PATH = MODELS_DIR / 'churn_pipeline.pkl'
THRESHOLD_PATH = MODELS_DIR / 'optimal_threshold.json'

# Target and Experimentation
TARGET = 'Churn'
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Cost Constants (Business Cost Matrix in USD)
COST_FALSE_NEGATIVE = 500  # Cost of missed churning customer (lost customer LTV)
COST_FALSE_POSITIVE = 50   # Cost of unnecessary retention incentive/discount