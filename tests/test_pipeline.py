import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from src.pipeline import build_pipeline, build_preprocessor
from src.preprocessing import FeatureEngineer, OutlierClipper


def get_sample_df() -> tuple[pd.DataFrame, pd.Series]:
    """Helper to return feature DataFrame and target Series."""
    df = pd.DataFrame(
        {
            "Tenure": [1.0, 5.0, np.nan, 20.0, 30.0, 2.0, np.nan, 15.0],
            "WarehouseToHome": [10.0, 30.0, 15.0, np.nan, 120.0, 8.0, 25.0, 12.0],
            "NumberOfDeviceRegistered": [3, 4, 2, 4, 5, 3, 4, 3],
            "PreferedOrderCat": [
                "Mobile",
                "Mobile Phone",
                "Laptop & Accessory",
                "Fashion",
                "Grocery",
                "Mobile",
                "Others",
                "Fashion",
            ],
            "SatisfactionScore": [1, 5, 3, 2, 4, 1, 2, 4],
            "MaritalStatus": [
                "Single",
                "Married",
                "Divorced",
                "Single",
                "Married",
                "Single",
                "Divorced",
                "Married",
            ],
            "NumberOfAddress": [2, 3, 1, 10, 25, 4, 2, 3],
            "Complain": [1, 0, 0, 1, 0, 1, 0, 0],
            "DaySinceLastOrder": [2.0, 7.0, np.nan, 1.0, 14.0, 3.0, 5.0, 0.0],
            "CashbackAmount": [120.5, 180.0, 150.2, 110.0, 220.0, 130.0, 160.0, 190.5],
            "Churn": [1, 0, 0, 1, 0, 1, 0, 0],
        }
    )
    return df.drop(columns=["Churn"]), df["Churn"]


class TestBuildPreprocessor:
    def test_preprocessor_structure(self):
        preprocessor = build_preprocessor(scale=True)
        transformer_names = [name for name, _, _ in preprocessor.transformers]
        assert transformer_names == ["num", "cat", "bin"]

    def test_preprocessor_output_pandas(self):
        X, _ = get_sample_df()
        fe = FeatureEngineer()
        clipper = OutlierClipper()

        df_fe = fe.fit_transform(clipper.fit_transform(X))
        preprocessor = build_preprocessor(scale=True)
        transformed = preprocessor.fit_transform(df_fe)

        assert isinstance(transformed, pd.DataFrame)
        assert not transformed.isna().any().any()

    def test_preprocessor_handles_unseen_categories(self):
        X, _ = get_sample_df()
        fe = FeatureEngineer()
        clipper = OutlierClipper()

        df_fe = fe.fit_transform(clipper.fit_transform(X))
        preprocessor = build_preprocessor(scale=False)
        preprocessor.fit(df_fe)

        unseen_df = df_fe.copy()
        unseen_df.loc[0, "PreferedOrderCat"] = "CompletelyUnseenCategory"
        unseen_df.loc[0, "MaritalStatus"] = "UnknownStatus"

        transformed = preprocessor.transform(unseen_df)
        assert isinstance(transformed, pd.DataFrame)
        assert not transformed.isna().any().any()


class TestBuildPipeline:
    def test_pipeline_steps(self):
        pipe = build_pipeline(DummyClassifier())
        step_names = [name for name, _ in pipe.steps]
        assert step_names == ["clip", "fe", "pre", "model"]

    def test_pipeline_fit_predict_proba(self):
        X, y = get_sample_df()

        pipe = build_pipeline(
            model=LogisticRegression(solver="liblinear", random_state=42),
            scale=True,
        )
        pipe.fit(X, y)

        proba = pipe.predict_proba(X)
        assert proba.shape == (len(X), 2)
        assert (proba >= 0.0).all() and (proba <= 1.0).all()
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)

        preds = pipe.predict(X)
        assert len(preds) == len(X)
        assert set(preds).issubset({0, 1})
