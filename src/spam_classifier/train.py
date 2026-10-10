"""Model training pipeline for Spam Message Classifier.

Performs:
1. Data loading and stratified train/test split (fixed random seed = 42).
2. Prevents data leakage by fitting TF-IDF strictly on the training partition.
3. Trains Logistic Regression (primary baseline) and Multinomial Naive Bayes (comparison).
4. Evaluates both on the held-out test set.
5. Saves the winning pipeline and metadata to `models/`.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from src.spam_classifier.data_loader import load_raw_data
from src.spam_classifier.preprocessing import clean_text

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

RANDOM_STATE = 42
TEST_SIZE = 0.20


def build_pipeline(classifier_type: str = "logistic_regression") -> Pipeline:
    """Construct an end-to-end Pipeline with TF-IDF and a chosen classifier.

    Args:
        classifier_type: 'logistic_regression' or 'naive_bayes'.

    Returns:
        Configured scikit-learn Pipeline.
    """
    vectorizer = TfidfVectorizer(
        preprocessor=clean_text,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )

    if classifier_type == "logistic_regression":
        classifier = LogisticRegression(
            C=5.0,
            max_iter=1000,
            random_state=RANDOM_STATE,
            solver="liblinear",
        )
    elif classifier_type == "naive_bayes":
        classifier = MultinomialNB(alpha=0.1)
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}")

    return Pipeline(steps=[("tfidf", vectorizer), ("classifier", classifier)])


def evaluate_model(pipeline: Pipeline, X_test: pd.Series, y_test: pd.Series) -> Dict[str, float]:
    """Calculate key performance metrics on held-out test data.

    In spam detection:
    - Precision measures: Out of all messages flagged as spam, how many were actually spam?
      (Higher precision means fewer legitimate emails sent to the spam folder).
    - Recall measures: Out of all actual spam messages, how many were caught?
    """
    y_pred = pipeline.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_spam": float(precision_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "recall_spam": float(recall_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "f1_spam": float(f1_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
    }


def train_and_select_model() -> Tuple[Pipeline, str, Dict[str, dict]]:
    """Train models, compare metrics, and persist the top-performing pipeline."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading raw dataset...")
    df = load_raw_data()

    # Stratified Train/Test split ensures both splits have the same spam proportion (~13.4%)
    print(f"Splitting dataset: 80% Train, 20% Held-Out Test (stratified, random_state={RANDOM_STATE})...")
    X_train, X_test, y_train, y_test = train_test_split(
        df["message"],
        df["label_num"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["label_num"],
    )

    # Persist the held-out test partition for evaluation and reproducible validation
    test_df = pd.DataFrame({"message": X_test, "label_num": y_test})
    test_df.to_csv(PROCESSED_DIR / "test_split.csv", index=False, encoding="utf-8")

    models_to_train = {
        "Logistic Regression": build_pipeline("logistic_regression"),
        "Multinomial Naive Bayes": build_pipeline("naive_bayes"),
    }

    results = {}
    comparison_rows = []

    for name, pipe in models_to_train.items():
        print(f"Training {name}...")
        pipe.fit(X_train, y_train)
        metrics = evaluate_model(pipe, X_test, y_test)
        results[name] = {"pipeline": pipe, "metrics": metrics}
        comparison_rows.append({"Model": name, **metrics})
        print(
            f"  {name} -> Acc: {metrics['accuracy']:.4f} | "
            f"Precision (Spam): {metrics['precision_spam']:.4f} | "
            f"Recall (Spam): {metrics['recall_spam']:.4f} | "
            f"F1 (Spam): {metrics['f1_spam']:.4f}"
        )

    # Save model comparison table
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_path = REPORTS_DIR / "model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print(f"Saved model comparison table to: {comparison_path}")

    # Select the model with highest F1 score on Spam class
    best_model_name = max(results.keys(), key=lambda m: results[m]["metrics"]["f1_spam"])
    best_pipe = results[best_model_name]["pipeline"]
    best_metrics = results[best_model_name]["metrics"]

    print(f"\nWinning Model Selected: {best_model_name} (F1-Spam: {best_metrics['f1_spam']:.4f})")

    # Persist the fitted pipeline artifact
    model_artifact_path = MODELS_DIR / "spam_classifier_pipeline.joblib"
    joblib.dump(best_pipe, model_artifact_path)
    print(f"Saved model artifact to: {model_artifact_path}")

    # Vocabulary details
    tfidf_step = best_pipe.named_steps["tfidf"]
    vocab_size = len(tfidf_step.vocabulary_)

    # Save metadata
    metadata = {
        "model_name": best_model_name,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "vocabulary_size": vocab_size,
        "metrics_on_test_split": best_metrics,
        "labels": {"0": "Not Spam (Ham)", "1": "Spam"},
    }
    metadata_path = MODELS_DIR / "model_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved model metadata to: {metadata_path}")

    return best_pipe, best_model_name, results


def main() -> None:
    """Entry point for training workflow."""
    train_and_select_model()


if __name__ == "__main__":
    main()
