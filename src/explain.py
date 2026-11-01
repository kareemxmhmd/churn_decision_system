import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.inspection import permutation_importance

from src import config


def load_model_and_test_data() -> tuple[BaseEstimator, pd.DataFrame, pd.Series]:
    model: BaseEstimator = joblib.load(config.MODEL_PATH)
    test_df = pd.read_csv(config.TEST_DATA)

    X_test = test_df.drop(columns=[config.TARGET])
    y_test = test_df[config.TARGET]
    return model, X_test, y_test


def extract_xgb_feature_importance(model: BaseEstimator) -> pd.DataFrame:
    fold_importances: list[np.ndarray] = []
    feature_names: list[str] = []

    for cc in model.calibrated_classifiers_:
        pipe = cc.estimator
        preprocessor = pipe.named_steps["pre"]
        xgb_model = pipe.named_steps["model"]

        if not feature_names:
            feature_names = list(preprocessor.get_feature_names_out())

        fold_importances.append(xgb_model.feature_importances_)

    importances_matrix = np.vstack(fold_importances)
    df_imp = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_mean": importances_matrix.mean(axis=0),
            "importance_std": importances_matrix.std(axis=0),
        }
    ).sort_values("importance_mean", ascending=False).reset_index(drop=True)

    return df_imp


def compute_permutation_importance(
    model: BaseEstimator, X_test: pd.DataFrame, y_test: pd.Series
) -> pd.DataFrame:

    perm = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="average_precision",
        n_repeats=10,
        random_state=42,
        n_jobs=-1,
    )

    df_perm = pd.DataFrame(
        {
            "feature": list(X_test.columns),
            "pr_auc_drop_mean": perm.importances_mean,
            "pr_auc_drop_std": perm.importances_std,
        }
    ).sort_values("pr_auc_drop_mean", ascending=False).reset_index(drop=True)

    return df_perm


def plot_feature_importances(
    xgb_df: pd.DataFrame,
    perm_df: pd.DataFrame,
    output_path: Path,
    top_n: int = 12,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    # 1. Plot XGBoost Engineered Feature Importance
    top_xgb = xgb_df.head(top_n).iloc[::-1]
    ax = axes[0]
    ax.barh(
        top_xgb["feature"],
        top_xgb["importance_mean"],
        xerr=top_xgb["importance_std"],
        color="steelblue",
        alpha=0.85,
         capsize=3,
    )
    ax.set_xlabel("Mean Gain Importance (across 5 CV folds)")
    ax.set_title(f"Top {top_n} Engineered Features (XGBoost Gain)")
    ax.grid(axis="x", alpha=0.3)

    # 2. Plot Permutation Importance on Raw Features
    top_perm = perm_df.head(top_n).iloc[::-1]
    ax = axes[1]
    ax.barh(
        top_perm["feature"],
        top_perm["pr_auc_drop_mean"],
        xerr=top_perm["pr_auc_drop_std"],
        color="darkorange",
        alpha=0.85,
        capsize=3,
    )
    ax.set_xlabel("Mean Decrease in Test PR-AUC")
    ax.set_title("Raw Feature Impact (Permutation Importance)")
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    model, X_test, y_test = load_model_and_test_data()

    xgb_df = extract_xgb_feature_importance(model)
    perm_df = compute_permutation_importance(model, X_test, y_test)

    print("Top 10 Engineered Features (XGBoost Gain):")
    print(xgb_df.head(10).to_string(index=False))
    print("\nRaw Features Permutation Importance (Test PR-AUC Drop):")
    print(perm_df.to_string(index=False))

    # Save JSON report
    report: dict[str, Any] = {
        "xgboost_engineered_importance": xgb_df.to_dict(orient="records"),
        "permutation_raw_importance": perm_df.to_dict(orient="records"),
    }
    importance_json_path = config.PROCESSED_DIR / "feature_importance.json"
    importance_json_path.parent.mkdir(parents=True, exist_ok=True)
    importance_json_path.write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    # Save Plot
    plot_path = config.PLOTS_DIR / "feature_importance.png"
    plot_feature_importances(xgb_df, perm_df, plot_path)

    print(f"\nSaved importance report to: {importance_json_path}")
    print(f"Saved importance plot to:   {plot_path}")


if __name__ == "__main__":
    main()