from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from src import config
from src.inference import load_inference_artifacts, predict_churn
from src.pipeline import build_pipeline


def get_single_customer() -> dict:
    return {
        "Tenure": 2.0,
        "WarehouseToHome": 15.0,
        "NumberOfDeviceRegistered": 3,
        "PreferedOrderCat": "Mobile Phone",
        "SatisfactionScore": 1,
        "MaritalStatus": "Single",
        "NumberOfAddress": 4,
        "Complain": 1,
        "DaySinceLastOrder": 2.0,
        "CashbackAmount": 130.0,
    }


def get_customer_list() -> list[dict]:
    cust1 = get_single_customer()
    cust2 = cust1.copy()
    cust2.update({"Tenure": 25.0, "Complain": 0, "MaritalStatus": "Married"})
    return [cust1, cust2]


def get_dummy_fitted_pipeline():
    X = pd.DataFrame([get_single_customer(), get_single_customer()])
    y = pd.Series([1, 0])
    pipe = build_pipeline(
        model=DummyClassifier(strategy="constant", constant=1),
        scale=False,
    )
    pipe.fit(X, y)
    return pipe


class TestInferenceArtifacts:
    def test_load_inference_artifacts_existing(self):
        if config.MODEL_PATH.exists():
            model, threshold = load_inference_artifacts()
            assert model is not None
            assert isinstance(threshold, float)
            assert 0.0 <= threshold <= 1.0

    def test_load_inference_artifacts_missing_file_raises(self, monkeypatch):
        fake_path = Path("non_existent_dir/fake_model.pkl")
        monkeypatch.setattr(config, "MODEL_PATH", fake_path)

        with pytest.raises(FileNotFoundError, match="Model file not found"):
            load_inference_artifacts()


class TestPredictChurn:
    def test_predict_with_single_dict(self):
        pipe = get_dummy_fitted_pipeline()
        cust = get_single_customer()

        res = predict_churn(cust, model=pipe, threshold=0.5)
        assert isinstance(res, pd.DataFrame)
        assert len(res) == 1
        assert list(res.columns) == [
            "churn_probability",
            "churn_prediction",
            "threshold_used",
        ]
        assert res.loc[0, "threshold_used"] == 0.5
        assert res.loc[0, "churn_prediction"] in [0, 1]

    def test_predict_with_list_of_dicts(self):
        pipe = get_dummy_fitted_pipeline()
        cust_list = get_customer_list()

        res = predict_churn(cust_list, model=pipe, threshold=0.3)
        assert isinstance(res, pd.DataFrame)
        assert len(res) == len(cust_list)
        assert (res["threshold_used"] == 0.3).all()

    def test_predict_with_dataframe_and_index_preservation(self):
        pipe = get_dummy_fitted_pipeline()
        df = pd.DataFrame(get_customer_list(), index=["user_alpha", "user_beta"])

        res = predict_churn(df, model=pipe, threshold=0.25)
        assert isinstance(res, pd.DataFrame)
        assert len(res) == len(df)
        assert list(res.index) == ["user_alpha", "user_beta"]

    def test_predict_drops_target_column_if_present(self):
        pipe = get_dummy_fitted_pipeline()
        df = pd.DataFrame(get_customer_list())
        df[config.TARGET] = [1, 0]

        res = predict_churn(df, model=pipe, threshold=0.5)
        assert isinstance(res, pd.DataFrame)
        assert len(res) == len(df)

    def test_invalid_data_type_raises_type_error(self):
        pipe = get_dummy_fitted_pipeline()

        with pytest.raises(TypeError, match="Input data must be a pandas DataFrame"):
            predict_churn("invalid_string_input", model=pipe, threshold=0.5)

        with pytest.raises(TypeError, match="Input data must be a pandas DataFrame"):
            predict_churn(12345, model=pipe, threshold=0.5)

    def test_prediction_threshold_logic(self):
        pipe = get_dummy_fitted_pipeline()
        cust = get_single_customer()

        res_high = predict_churn(cust, model=pipe, threshold=0.9999)
        proba = res_high.loc[0, "churn_probability"]
        pred = res_high.loc[0, "churn_prediction"]
        assert pred == int(proba >= 0.9999)
