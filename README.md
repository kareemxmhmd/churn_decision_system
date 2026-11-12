# Customer Churn Decision System

A production-ready ML decision system that turns calibrated churn probabilities into cost-optimal customer retention actions.

**Live App:** [Churn Decision System (Streamlit)](https://churndecisionsystem-s98.streamlit.app)

---

## 1. Business Focus

- **Problem:** Acquiring a new customer is far more expensive than retaining an existing one, but sending retention incentives to everyone wastes budget.
- **Cost-Aware Decision Matrix:**
  - **Missed Churn (False Negative):** **\$500** lost customer lifetime value (LTV).
  - **Unnecessary Incentive (False Positive):** **\$50** retention offer cost (10:1 asymmetric cost).
- **Impact & Results (Test Set):**
  - Optimizing the decision threshold with an **$F_2$-score** ($\beta = 2.0$, prioritizing recall) on out-of-fold calibrated probabilities shifts the operating threshold from `0.50` to **`0.1515`**.
  - **Recall** improves from **79.4%** to **92.5%** (`PR-AUC = 0.808`, `ROC-AUC = 0.960`, `Brier = 0.059`).
  - **Total Business Cost** drops from **\$12,500** (default `0.50`) to **\$7,950** — a **36.4% cost reduction** on the held-out test set.

---

## 2. Software & ML Engineering Execution

- **Leakage-Free ML Pipeline:** Built custom scikit-learn transformers (`OutlierClipper`, `FeatureEngineer`) inside a unified `Pipeline` (`src/preprocessing.py`, `src/pipeline.py`) paired with 5-fold stratified CV and sigmoid probability calibration (`CalibratedClassifierCV`) in `src/train.py`.
- **Cost-Driven Evaluation & Explainability:** Separated out-of-fold threshold tuning from final test evaluation (`src/evaluate.py`), alongside XGBoost feature importance and permutation importance (`src/explain.py`).
- **Experiment Tracking:** Integrated MLflow (`mlflow.db`) to log parameters, CV/test metrics, diagnostic plots, and serialized model artifacts (`models/churn_pipeline.pkl`, `models/threshold.json`).
- **Production Serving & UI:** Implemented a Pydantic-validated FastAPI service (`api/main.py` supporting single `/predict`, `/predict/batch`, and `/health`), a Streamlit app (`app/main.py`), and a lightweight HTML client (`HTML/index.html`), containerized via `Dockerfile` and `docker-compose.yml`.
- **Quality Assurance & CI/CD:** Wrote 32 unit, pipeline, inference, and API tests (`tests/`) with `pytest`, enforced code checks with `ruff`, and automated testing + Docker image verification in GitHub Actions (`.github/workflows/ci.yml`), with initial PSI drift monitoring started in `src/monitor.py`.
