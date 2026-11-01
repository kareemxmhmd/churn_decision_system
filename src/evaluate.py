import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src import config


def load_artifacts() -> tuple[BaseEstimator, pd.DataFrame, pd.DataFrame, pd.Series]:
    model: BaseEstimator = joblib.load(config.MODEL_PATH)
    oof_df = pd.read_csv(config.OOF_TRAIN_PATH)
    test_df = pd.read_csv(config.TEST_DATA)

    X_test = test_df.drop(columns=[config.TARGET])
    y_test = test_df[config.TARGET]
    return model, oof_df, X_test, y_test


def find_optimal_threshold(
    y_true: np.ndarray,
    proba: np.ndarray,
    beta: float = 1.0,
) -> dict[str, float]:
    
    precisions, recalls, thresholds = precision_recall_curve(y_true, proba)

    prec = precisions[:-1]
    rec = recalls[:-1]

    beta_sq = beta**2
    denom = (beta_sq * prec) + rec
    f_beta = np.where(denom > 0, (1 + beta_sq) * (prec * rec) / denom, 0.0)

    best_idx = int(np.argmax(f_beta))
    return {
        "threshold": float(thresholds[best_idx]),
        "beta": float(beta),
        "oof_f_beta": float(f_beta[best_idx]),
        "oof_precision": float(prec[best_idx]),
        "oof_recall": float(rec[best_idx]),
    }


def evaluate_at_threshold(
    y_true: np.ndarray,
    proba: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    preds = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()

    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    total_cost = int(
    (fn * config.COST_FALSE_NEGATIVE) + (fp * config.COST_FALSE_POSITIVE)
    )

    return {
        "threshold": float(threshold),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "brier": float(brier_score_loss(y_true, proba)),
        "baseline_pr_auc": float(np.mean(y_true)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "specificity": specificity,
        "predicted_positive_rate": float(np.mean(preds)),
        "business_cost_usd": total_cost,
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


def plot_diagnostics(
    y_test: np.ndarray,
    test_proba: np.ndarray,
    threshold: float,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    prec, rec, _ = precision_recall_curve(y_test, test_proba)
    pr_auc = average_precision_score(y_test, test_proba)
    baseline = float(np.mean(y_test))
    preds_opt = (test_proba >= threshold).astype(int)
    opt_prec = precision_score(y_test, preds_opt, zero_division=0)
    opt_rec = recall_score(y_test, preds_opt, zero_division=0)

    ax = axes[0]
    ax.plot(rec, prec, lw=2, label=f"Model (PR-AUC = {pr_auc:.3f})")
    ax.axhline(baseline, color="gray", linestyle="--", label=f"Baseline ({baseline:.3f})")
    ax.scatter(
        [opt_rec],
        [opt_prec],
        color="crimson",
        zorder=5,
        label=f"Threshold = {threshold:.3f}",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve (Test)")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)

    fpr, tpr, _ = roc_curve(y_test, test_proba)
    roc_auc = roc_auc_score(y_test, test_proba)
    tn, fp, fn, tp = confusion_matrix(y_test, preds_opt, labels=[0, 1]).ravel()
    opt_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    ax = axes[1]
    ax.plot(fpr, tpr, lw=2, label=f"Model (ROC-AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random")
    ax.scatter(
        [opt_fpr],
        [opt_rec],
        color="crimson",
        zorder=5,
        label=f"Threshold = {threshold:.3f}",
    )
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title("ROC Curve (Test)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    prob_true, prob_pred = calibration_curve(y_test, test_proba, n_bins=10, strategy="quantile")
    brier = brier_score_loss(y_test, test_proba)

    ax = axes[2]
    ax.plot(prob_pred, prob_true, marker="o", lw=2, label=f"Calibrated (Brier = {brier:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Perfectly Calibrated")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curve (Test)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def print_summary(label: str, metrics: dict[str, Any]) -> None:
    cm = metrics["confusion_matrix"]
    print(f"{label} (threshold = {metrics['threshold']:.4f})")

    print(f"PR-AUC: {metrics['pr_auc']:.3f} | ROC-AUC: {metrics['roc_auc']:.3f} | "
          f"Brier: {metrics['brier']:.4f}")
    
    print(f"F1:     {metrics['f1']:.3f} | Precision: {metrics['precision']:.3f} | "
          f"Recall: {metrics['recall']:.3f} | Specificity: {metrics['specificity']:.3f}")
    
    print(f"Confusion Matrix -> TN: {cm['tn']:,}, FP: {cm['fp']:,}, "
          f"FN: {cm['fn']:,}, TP: {cm['tp']:,}")
    
    print(f"Cost: ${metrics['business_cost_usd']:,}\n")


def main() -> None:
    model, oof_df, X_test, y_test = load_artifacts()

    oof_y = oof_df["y_true"].to_numpy()
    oof_proba = oof_df["proba"].to_numpy()

    y_test_arr = y_test.to_numpy()
    test_proba = model.predict_proba(X_test)[:, 1]

    f_beta: float = float(getattr(config, "F_BETA", 1.0))
    opt_info = find_optimal_threshold(oof_y, oof_proba, beta=f_beta)
    best_threshold = opt_info["threshold"]

    oof_eval_opt = evaluate_at_threshold(oof_y, oof_proba, best_threshold)
    test_eval_default = evaluate_at_threshold(y_test_arr, test_proba, 0.5)
    test_eval_opt = evaluate_at_threshold(y_test_arr, test_proba, best_threshold)

    print(
        f"Optimal OOF Threshold (F{f_beta:g}): {best_threshold:.4f} "
        f"(OOF F{f_beta:g} = {opt_info['oof_f_beta']:.3f})\n"
    )
    print_summary("OOF (Optimal Threshold)", oof_eval_opt)
    print_summary("Test (Default Threshold 0.50)", test_eval_default)
    print_summary("Test (Optimal OOF Threshold)", test_eval_opt)

    # Save or update metrics JSON
    existing_metrics: dict[str, Any] = {}
    if config.METRICS_PATH.exists():
        existing_metrics = json.loads(config.METRICS_PATH.read_text(encoding="utf-8"))

    existing_metrics["threshold_tuning"] = opt_info
    existing_metrics["oof_at_optimal_threshold"] = oof_eval_opt
    existing_metrics["test_at_default_0_5"] = test_eval_default
    existing_metrics["test_at_optimal_threshold"] = test_eval_opt

    config.METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.METRICS_PATH.write_text(
        json.dumps(existing_metrics, indent=2), encoding="utf-8"
    )

    threshold_path: Path = getattr(
        config,
        "THRESHOLD_PATH",
        config.MODEL_PATH.with_name("threshold.json"),
    )
    threshold_path.parent.mkdir(parents=True, exist_ok=True)
    threshold_path.write_text(
        json.dumps(opt_info, indent=2), encoding="utf-8"
    )

    plots_dir: Path = getattr(
        config,
        "PLOTS_DIR",
        config.METRICS_PATH.parent / "plots",
    )
    plot_path = plots_dir / "evaluation_diagnostics.png"
    plot_diagnostics(y_test_arr, test_proba, best_threshold, plot_path)

    print(f"Updated metrics saved to: {config.METRICS_PATH}")
    print(f"Threshold artifact saved to: {threshold_path}")
    print(f"Diagnostic plots saved to: {plot_path}")


if __name__ == "__main__":
    main()