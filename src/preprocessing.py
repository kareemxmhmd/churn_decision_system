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

class OutlierClipper(BaseEstimator, TransformerMixin):

    def __init__(
               self,
               columns: Optional[List[Union[str]]] = None,
               upper_quantile: float = 0.99,
     ) -> None:
          self.columns = columns
          self.upper_quantile = upper_quantile

    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> "OutlierClipper":
        df = _ensure_dataframe(X)

        if self.columns is None:
              self.columns_: List[Union[str]] = ["WarehouseToHome", "NumberOfAddress"]
        else:
             self.columns_ = list(self.columns)

        missing = [col for col in self.columns_ if col not in df.columns]
        if missing:
             raise ValueError(f"Missing required columns for clipping: {missing}")

        self.upper_limits_: Dict[Union[str, int], float] = {
             col: float(df[col].quantile(self.upper_quantile))
             for col in self.columns_
        }
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = _ensure_dataframe(X)

        missing = [col for col in self.columns_ if col not in df.columns]
        if missing:
             raise ValueError(f"Missing required columns for clipping: {missing}")

        for col in self.columns_:
             df[col] = df[col].clip(upper=self.upper_limits_[col])

        return df
