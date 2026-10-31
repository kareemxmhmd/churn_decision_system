from typing import Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


def _ensure_dataframe(X: object, copy: bool = True) -> pd.DataFrame:
    if not isinstance(X, pd.DataFrame):
        raise TypeError(
            f"Expected a pandas DataFrame with named columns, got {type(X).__name__}. "
            "If this step follows other pipeline steps, call pipe.set_output(transform='pandas')."
        )
    return X.copy() if copy else X


class OutlierClipper(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        columns: Optional[list[str]] = None,
        upper_quantile: float = 0.99,
    ) -> None:
        self.columns = columns
        self.upper_quantile = upper_quantile

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "OutlierClipper":
        df = _ensure_dataframe(X, copy=False)

        if self.columns is None:
            self.columns_: list[str] = ["WarehouseToHome", "NumberOfAddress"]
        else:
            self.columns_ = list(self.columns)

        missing = [col for col in self.columns_ if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns for clipping: {missing}")

        self.upper_limits_: dict[str, float] = {
            col: float(df[col].quantile(self.upper_quantile))
            for col in self.columns_
        }
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, ["columns_", "upper_limits_"])
        df = _ensure_dataframe(X, copy=True)

        missing = [col for col in self.columns_ if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns for clipping: {missing}")

        for col in self.columns_:
            df[col] = df[col].clip(upper=self.upper_limits_[col])

        return df


class FeatureEngineer(BaseEstimator, TransformerMixin):
    REQUIRED_COLUMNS: list[str] = [
        "Tenure",
        "WarehouseToHome",
        "PreferedOrderCat",
        "MaritalStatus",
        "Complain",
    ]

    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> "FeatureEngineer":
        df = _ensure_dataframe(X, copy=False)
        self._check_columns(df)
        self.n_features_in_: int = df.shape[1]
        return self

    def _check_columns(self, X: pd.DataFrame) -> None:
        missing = [col for col in self.REQUIRED_COLUMNS if col not in X.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, ["n_features_in_"])
        df = _ensure_dataframe(X, copy=True)
        self._check_columns(df)

        df["PreferedOrderCat"] = df["PreferedOrderCat"].replace(
            {"Mobile": "Mobile Phone"}
        )

        df["tenure_isna"] = df["Tenure"].isna().astype(int)
        df["warehouse_to_home_isna"] = df["WarehouseToHome"].isna().astype(int)

        tenure_binned = pd.cut(
            df["Tenure"],
            bins=[-np.inf, 3, 6, 12, 24, np.inf],
            labels=["0-3 mo", "4-6 mo", "7-12 mo", "13-24 mo", "25+ mo"],
            right=True,
        )
        df["tenure_band"] = (
            tenure_binned.cat.add_categories("Missing")
            .fillna("Missing")
            .astype(str)
        )

        df["complain_x_low_tenure"] = (
            (df["Complain"] == 1) & (df["Tenure"].le(3, fill_value=np.inf))
        ).astype(int)
        df["complain_x_single"] = (
            (df["Complain"] == 1) & (df["MaritalStatus"] == "Single")
        ).astype(int)

        return df