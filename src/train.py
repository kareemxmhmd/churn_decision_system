import json
from typing import Any

import joblib
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src import config
from src.pipeline import build_pipeline


def load_data() -> pd.DataFrame:
    df = pd.read_csv(config.RAW_DATA)
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)

    print(f"Rows before dropping duplicates: {n_before:,}")
    print(f"Rows after dropping duplicates:  {len(df):,}")
    return df


def split_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = df.drop(columns=[config.TARGET])
    y = df[config.TARGET]

    return train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )


def build_model(y_train: pd.Series) -> Pipeline:
    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    scale_pos_weight = neg_count / pos_count

    return build_pipeline(
        XGBClassifier(
            n_estimators=400,
            learning_rate=0.03,
            max_depth=4,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.5,
            reg_lambda=2.0,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=42,
            n_jobs=1,
        ),
        scale=False,
    )


def make_cv() -> StratifiedKFold:
    return StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )


def cross_validate_model(
    pipe: Pipeline, X_train: pd.DataFrame, y_train: pd.Series
) -> dict[str, float]:
    scores = cross_validate(
        pipe,
        X_train,
        y_train,
        cv=make_cv(),
        scoring={"pr_auc": "average_precision", "roc_auc": "roc_auc"},
        n_jobs=-1,
        error_score="raise",
    )

    result = {
        "pr_auc_mean": float(scores["test_pr_auc"].mean()),
        "pr_auc_std": float(scores["test_pr_auc"].std()),
        "roc_auc_mean": float(scores["test_roc_auc"].mean()),
        "roc_auc_std": float(scores["test_roc_auc"].std()),
    }

    print(
        f"CV PR-AUC: {result['pr_auc_mean']:.3f} (+/- {result['pr_auc_std']:.3f}) | "
        f"ROC-AUC: {result['roc_auc_mean']:.3f} (+/- {result['roc_auc_std']:.3f})"
    )
    return result


def calibrate(pipe: Pipeline) -> CalibratedClassifierCV:
    return CalibratedClassifierCV(
        estimator=pipe, method="sigmoid", cv=make_cv(), n_jobs=1
    )


def evaluate(
    model: BaseEstimator, X: pd.DataFrame, y: pd.Series
) -> dict[str, float]:
    proba = model.predict_proba(X)[:, 1]
    return {
        "pr_auc": float(average_precision_score(y, proba)),
        "roc_auc": float(roc_auc_score(y, proba)),
        "brier": float(brier_score_loss(y, proba)),
        "baseline_pr_auc": float(y.mean()),
    }


def main() -> None:
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"Train: {len(X_train):,} | Test: {len(X_test):,}\n")

    cv_results = cross_validate_model(build_model(y_train), X_train, y_train)

    uncalibrated = build_model(y_train).fit(X_train, y_train)
    calibrated = calibrate(build_model(y_train)).fit(X_train, y_train)

    oof = cross_val_predict(
    calibrate(build_model(y_train)),
    X_train,
    y_train,
    cv=make_cv(),
    method="predict_proba",
    n_jobs=-1,
    )[:, 1]

    metrics: dict[str, Any] = {
        "model": "xgboost",
        "cv": cv_results,
        "oof_calibrated": {
            "pr_auc": float(average_precision_score(y_train, oof)),
            "roc_auc": float(roc_auc_score(y_train, oof)),
            "brier": float(brier_score_loss(y_train, oof)),
        },
        "test_uncalibrated": evaluate(uncalibrated, X_test, y_test),
        "test_calibrated": evaluate(calibrated, X_test, y_test),
    }

    config.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(calibrated, config.MODEL_PATH, compress=3)

    pd.DataFrame({"y_true": y_train.to_numpy(), "proba": oof}).to_csv(
        config.OOF_TRAIN_PATH, index=False
    )

    X_test.assign(**{config.TARGET: y_test}).to_csv(
        config.TEST_DATA, index=False
    )

    config.METRICS_PATH.write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(f"\nSaved calibrated model to: {config.MODEL_PATH}")


if __name__ == "__main__":
    main()