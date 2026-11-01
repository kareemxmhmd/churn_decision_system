import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from src.preprocessing import FeatureEngineer, OutlierClipper, _ensure_dataframe


def get_sample_features() -> pd.DataFrame:
    """Helper to generate a self-contained sample dataframe."""
    return pd.DataFrame(
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
        }
    )


class TestEnsureDataFrame:
    def test_valid_dataframe_copy(self):
        df = get_sample_features()
        df_copy = _ensure_dataframe(df, copy=True)
        assert isinstance(df_copy, pd.DataFrame)
        assert df_copy is not df
        assert df_copy.equals(df)

    def test_valid_dataframe_no_copy(self):
        df = get_sample_features()
        df_ref = _ensure_dataframe(df, copy=False)
        assert df_ref is df

    def test_invalid_type_raises_type_error(self):
        with pytest.raises(TypeError, match="Expected a pandas DataFrame"):
            _ensure_dataframe([{"a": 1}, {"a": 2}])

        with pytest.raises(TypeError, match="Expected a pandas DataFrame"):
            _ensure_dataframe(np.array([[1, 2], [3, 4]]))


class TestOutlierClipper:
    def test_fit_calculates_upper_limits_default(self):
        df = get_sample_features()
        clipper = OutlierClipper(upper_quantile=0.90)
        clipper.fit(df)

        assert hasattr(clipper, "upper_limits_")
        assert "WarehouseToHome" in clipper.upper_limits_
        assert "NumberOfAddress" in clipper.upper_limits_
        assert clipper.columns_ == ["WarehouseToHome", "NumberOfAddress"]

    def test_fit_custom_columns(self):
        df = get_sample_features()
        clipper = OutlierClipper(columns=["Tenure"], upper_quantile=0.50)
        clipper.fit(df)

        assert clipper.columns_ == ["Tenure"]
        assert "Tenure" in clipper.upper_limits_

    def test_transform_clips_values(self):
        df = pd.DataFrame(
            {
                "WarehouseToHome": [10.0, 20.0, 30.0, 100.0],
                "NumberOfAddress": [1, 2, 3, 20],
            }
        )
        clipper = OutlierClipper(upper_quantile=0.50)
        transformed = clipper.fit_transform(df)

        assert transformed["WarehouseToHome"].max() <= 25.0
        assert transformed["NumberOfAddress"].max() <= 2.5
        assert transformed.loc[0, "WarehouseToHome"] == 10.0

    def test_missing_column_raises_error(self):
        df = get_sample_features().drop(columns=["WarehouseToHome"])
        clipper = OutlierClipper()

        with pytest.raises(ValueError, match="Missing required columns"):
            clipper.fit(df)

    def test_not_fitted_raises_error(self):
        df = get_sample_features()
        clipper = OutlierClipper()
        with pytest.raises(NotFittedError):
            clipper.transform(df)


class TestFeatureEngineer:
    def test_transform_creates_expected_columns(self):
        df = get_sample_features()
        fe = FeatureEngineer()
        transformed = fe.fit_transform(df)

        expected_new_cols = [
            "tenure_isna",
            "warehouse_to_home_isna",
            "tenure_band",
            "complain_x_low_tenure",
            "complain_x_single",
        ]
        for col in expected_new_cols:
            assert col in transformed.columns

    def test_prefered_order_cat_replacement(self):
        df = pd.DataFrame(
            {
                "Tenure": [1.0, 5.0],
                "WarehouseToHome": [10.0, 15.0],
                "PreferedOrderCat": ["Mobile", "Mobile Phone"],
                "MaritalStatus": ["Single", "Married"],
                "Complain": [0, 1],
            }
        )
        fe = FeatureEngineer()
        transformed = fe.fit_transform(df)

        assert (transformed["PreferedOrderCat"] == "Mobile Phone").all()

    def test_missing_indicators(self):
        df = pd.DataFrame(
            {
                "Tenure": [np.nan, 5.0],
                "WarehouseToHome": [10.0, np.nan],
                "PreferedOrderCat": ["Mobile Phone", "Laptop"],
                "MaritalStatus": ["Single", "Married"],
                "Complain": [0, 0],
            }
        )
        fe = FeatureEngineer()
        transformed = fe.fit_transform(df)

        assert list(transformed["tenure_isna"]) == [1, 0]
        assert list(transformed["warehouse_to_home_isna"]) == [0, 1]

    def test_tenure_band_binning_and_missing(self):
        df = pd.DataFrame(
            {
                "Tenure": [1.0, 5.0, 10.0, 20.0, 30.0, np.nan],
                "WarehouseToHome": [10.0] * 6,
                "PreferedOrderCat": ["Mobile Phone"] * 6,
                "MaritalStatus": ["Single"] * 6,
                "Complain": [0] * 6,
            }
        )
        fe = FeatureEngineer()
        transformed = fe.fit_transform(df)

        expected_bands = [
            "0-3 mo",
            "4-6 mo",
            "7-12 mo",
            "13-24 mo",
            "25+ mo",
            "Missing",
        ]
        assert list(transformed["tenure_band"]) == expected_bands

    def test_interaction_features(self):
        df = pd.DataFrame(
            {
                "Tenure": [1.0, 10.0, np.nan, 2.0],
                "WarehouseToHome": [10.0] * 4,
                "PreferedOrderCat": ["Mobile Phone"] * 4,
                "MaritalStatus": ["Single", "Single", "Single", "Married"],
                "Complain": [1, 1, 1, 1],
            }
        )
        fe = FeatureEngineer()
        transformed = fe.fit_transform(df)

        assert list(transformed["complain_x_low_tenure"]) == [1, 0, 0, 1]
        assert list(transformed["complain_x_single"]) == [1, 1, 1, 0]

    def test_missing_required_column_raises_error(self):
        df_incomplete = pd.DataFrame(
            {
                "Tenure": [1.0],
                "WarehouseToHome": [10.0],
            }
        )
        fe = FeatureEngineer()
        with pytest.raises(ValueError, match="Missing columns"):
            fe.fit(df_incomplete)

    def test_not_fitted_raises_error(self):
        df = get_sample_features()
        fe = FeatureEngineer()
        with pytest.raises(NotFittedError):
            fe.transform(df)
