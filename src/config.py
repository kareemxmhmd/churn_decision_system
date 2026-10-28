from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = ROOT / 'data' / 'raw' / 'customer_churn.csv'
MODEL_PATH = ROOT / 'models' / 'churn_pipeline.pkl'

TARGET = 'Churn'
RANDOM_STATE = 42
TEST_SIZE = 0.2

COST_FALSE_NEGATIVE = 500
COST_FALSE_POSITIVE = 50