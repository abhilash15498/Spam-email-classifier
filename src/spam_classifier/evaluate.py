"""Evaluation module for Spam Message Classifier.

Loads the saved model pipeline and held-out test dataset, computes comprehensive
classification metrics, generates an annotated confusion matrix plot, and exports
evaluation reports.
"""

import json
from pathlib import Path
from typing import Dict

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_FILE = MODELS_DIR / "spam_classifier_pipeline.joblib"
TEST_DATA_FILE = PROCESSED_DIR / "test_split.csv"


def evaluate_pipeline(
    model_path: Path = MODEL_FILE,
    test_data_path: Path = TEST_DATA_FILE,
    save_plot: bool = True,
) -> Dict[str, object]:
    """Run evaluation on the held-out test partition using the persisted model artifact.

    Args:
        model_path: Path to the .joblib pipeline artifact.
        test_data_path: Path to the test_split.csv file.
        save_plot: Whether to save the confusion matrix plot.

    Returns:
        Dictionary of metrics, classification report, and confusion matrix breakdown.
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. Please run train.py first."
        )

    if not test_data_path.exists():
        raise FileNotFoundError(
            f"Held-out test split not found at {test_data_path}. Please run train.py first."
        )

    print(f"Loading model from: {model_path}")
    pipeline = joblib.load(model_path)

    print(f"Loading held-out test data from: {test_data_path}")
    test_df = pd.read_csv(test_data_path)
    X_test = test_df["message"].fillna("")
    y_test = test_df["label_num"]

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else None

    # Compute metrics
    acc = accuracy_score(y_test, y_pred)
    prec_spam = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    rec_spam = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    f1_spam = f1_score(y_test, y_pred, pos_label=1, zero_division=0)

    prec_ham = precision_score(y_test, y_pred, pos_label=0, zero_division=0)
    rec_ham = recall_score(y_test, y_pred, pos_label=0, zero_division=0)
    f1_ham = f1_score(y_test, y_pred, pos_label=0, zero_division=0)

    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")

    # Confusion matrix:
    # [[TN, FP],
    #  [FN, TP]]
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    report_dict = classification_report(
        y_test,
        y_pred,
        target_names=["Not Spam (Ham)", "Spam"],
        output_dict=True,
    )

    results = {
        "overall_accuracy": float(acc),
        "spam_class": {
            "precision": float(prec_spam),
            "recall": float(rec_spam),
            "f1_score": float(f1_spam),
            "support": int(tp + fn),
        },
        "ham_class": {
            "precision": float(prec_ham),
            "recall": float(rec_ham),
            "f1_score": float(f1_ham),
            "support": int(tn + fp),
        },
        "macro_avg_f1": float(macro_f1),
        "weighted_avg_f1": float(weighted_f1),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
        "detailed_classification_report": report_dict,
    }

    # Save metrics JSON
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_file = REPORTS_DIR / "evaluation_metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved evaluation metrics to: {metrics_file}")

    if save_plot:
        # Plot annotated confusion matrix
        fig, ax = plt.subplots(figsize=(6, 5))
        labels = [["TN\n" + str(tn), "FP (Type I)\n" + str(fp)],
                  ["FN (Type II)\n" + str(fn), "TP\n" + str(tp)]]
        
        sns.heatmap(
            cm,
            annot=labels,
            fmt="",
            cmap="Blues",
            cbar=False,
            xticklabels=["Not Spam (Ham)", "Spam"],
            yticklabels=["Not Spam (Ham)", "Spam"],
            annot_kws={"fontsize": 12, "fontweight": "bold"},
            ax=ax,
        )
        ax.set_title("Test Confusion Matrix (Held-out Set)", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Predicted Label", fontsize=11, fontweight="bold")
        ax.set_ylabel("True Label", fontsize=11, fontweight="bold")

        plot_path = REPORTS_DIR / "confusion_matrix.png"
        fig.savefig(plot_path, dpi=300)
        plt.close(fig)
        print(f"Saved confusion matrix chart to: {plot_path}")

    return results


def main() -> None:
    """Run model evaluation and print human-readable summary."""
    print("Evaluating saved model pipeline on held-out test data...")
    res = evaluate_pipeline()

    print("\n" + "=" * 50)
    print("HELD-OUT TEST SET EVALUATION RESULTS")
    print("=" * 50)
    print(f"Overall Accuracy:       {res['overall_accuracy'] * 100:.2f}%")
    print("\nSpam Metrics:")
    print(f"  Precision (Spam):     {res['spam_class']['precision'] * 100:.2f}%")
    print(f"  Recall (Spam):        {res['spam_class']['recall'] * 100:.2f}%")
    print(f"  F1-Score (Spam):      {res['spam_class']['f1_score'] * 100:.2f}%")
    print("\nLegitimate (Ham) Metrics:")
    print(f"  Precision (Ham):      {res['ham_class']['precision'] * 100:.2f}%")
    print(f"  Recall (Ham):         {res['ham_class']['recall'] * 100:.2f}%")
    print(f"  F1-Score (Ham):       {res['ham_class']['f1_score'] * 100:.2f}%")

    cm = res["confusion_matrix"]
    print("\nConfusion Matrix Breakdown:")
    print(f"  True Negatives (Legitimate correctly identified): {cm['true_negatives']}")
    print(f"  False Positives (Legitimate wrongly flagged):    {cm['false_positives']}  <-- Critical Type I Error")
    print(f"  False Negatives (Spam missed):                   {cm['false_negatives']}")
    print(f"  True Positives (Spam correctly caught):          {cm['true_positives']}")
    print("=" * 50)


if __name__ == "__main__":
    main()
