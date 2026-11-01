from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.preprocessing import FeatureEngineer, OutlierClipper

NUMERIC_COLS: list[str] = [
    "Tenure",
    "WarehouseToHome",
    "NumberOfDeviceRegistered",
    "SatisfactionScore",
    "NumberOfAddress",
    "DaySinceLastOrder",
    "CashbackAmount",
]

CATEGORICAL_COLS: list[str] = [
    "PreferedOrderCat",
    "MaritalStatus",
    "tenure_band",
]

BINARY_COLS: list[str] = [
    "Complain",
    "tenure_isna",
    "warehouse_to_home_isna",
    "complain_x_low_tenure",
    "complain_x_single",
]


def build_preprocessor(scale: bool = True, drop_first: bool = False) -> ColumnTransformer:
    numeric_steps: list[tuple[str, BaseEstimator]] = [
        ("imputer", SimpleImputer(strategy="median"))
    ]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe",OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    drop="if_binary" if drop_first else None,
                ),
            ),
        ]
    )

    binary_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_steps), NUMERIC_COLS),
            ("cat", categorical_pipeline, CATEGORICAL_COLS),
            ("bin", binary_pipeline, BINARY_COLS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    preprocessor.set_output(transform="pandas")
    return preprocessor


def build_pipeline(model: BaseEstimator, scale: bool = True, drop_first: bool = False) -> Pipeline:
    return Pipeline(
        steps=[
            ("clip", OutlierClipper()),
            ("fe", FeatureEngineer()),
            ("pre", build_preprocessor(scale=scale, drop_first=drop_first)),
            ("model", model),
        ]
    )