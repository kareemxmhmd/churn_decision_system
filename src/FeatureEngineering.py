import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Dict, List, Optional, Union


def _ensure_dataframe(X: object) -> pd.DataFrame:
     if not isinstance(X, pd.DataFrame):
          raise TypeError(
               f"Expected a pandas DataFrame with named columns, got {type(X).__name__}. "
               "If this step follows other pipeline steps, call pipe.set_output(transform='pandas')."
          )

     return X.copy()

class featureEngineer(BaseEstimator, TransformerMixin):

     REQUIRED_COLUMND: List[str] = [
        "Tenure",
        "WarehouseToHome",
        "PreferedOrderCat",
        "MaritalStatus",
        "Complain"
     ]

     def fit(
               self, X: pd.DataFrame, y: Optional[pd.Series] = None
     ) -> "featureEngineer":
          df = _ensure_dataframe(X)
          self._check_columns(df)
          return self

     def _check_columns(self, X: pd.DataFrame) -> None:
          missing = [col for col in self.REQUIRED_COLUMND if col not in X.columns]
          if missing:
               raise ValueError(f"Missing columns: {missing}")

     def transform(self, X: pd.DataFrame) -> pd.DataFrame:
          df = _ensure_dataframe(X)
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
               right=True
          )
          df["tenure_band"] = (
               tenure_binned.astype(object).fillna("Missing").astype(str)
          )

          df["complain_x_low_tenure"] = (
               (df["Complain"] == 1) & (df["Tenure"] <= 3)
          ).astype(int)
          df["complain_x_single"] =(
               (df["Complain"] == 1) & (df["MaritalStatus"]== "Single")
          ).astype(int)

          return df
