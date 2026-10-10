"""Automated tests for model training, pipeline behavior, and predictions."""

import tempfile
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.spam_classifier.train import build_pipeline
from src.spam_classifier.evaluate import evaluate_pipeline


@pytest.fixture
def synthetic_corpus():
    """Create a small synthetic dataset for isolated unit testing."""
    X = [
        "Hey bro, are we having coffee today?",
        "Can you send me the lecture slides please?",
        "Don't forget Mom's birthday dinner tomorrow.",
        "URGENT! You won £1000 cash prize! Call 0905000032 now to claim!",
        "WINNER! Claim your free gift voucher at http://scam-site.org right now!",
        "Double your money guaranteed! Send £500 immediately to earn bonus.",
    ]
    y = [0, 0, 0, 1, 1, 1]
    return pd.Series(X), pd.Series(y)


def test_build_pipeline_and_synthetic_training(synthetic_corpus):
    """Verify that a pipeline can be constructed, fitted, and predict on synthetic data."""
    X, y = synthetic_corpus
    pipe = build_pipeline("logistic_regression")
    assert isinstance(pipe, Pipeline)

    pipe.fit(X, y)
    preds = pipe.predict(["Hello are you free tomorrow?"])
    assert preds.shape == (1,)
    assert preds[0] in [0, 1]

    probas = pipe.predict_proba(["Hello are you free tomorrow?"])
    assert probas.shape == (1, 2)
    assert 0.0 <= probas[0, 0] <= 1.0
    assert 0.0 <= probas[0, 1] <= 1.0
    assert np.isclose(probas[0, 0] + probas[0, 1], 1.0)


def test_pipeline_serialization(synthetic_corpus):
    """Verify that the fitted pipeline can be saved to disk and reloaded without error."""
    X, y = synthetic_corpus
    pipe = build_pipeline("naive_bayes")
    pipe.fit(X, y)

    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        joblib.dump(pipe, tmp_path)
        loaded_pipe = joblib.load(tmp_path)
        pred = loaded_pipe.predict(["Win $5000 now!"])
        assert pred[0] in [0, 1]
    finally:
        tmp_path.unlink(missing_ok=True)


def test_evaluate_pipeline_missing_model_raises_error():
    """Verify that evaluate_pipeline raises FileNotFoundError when model is missing."""
    with pytest.raises(FileNotFoundError):
        evaluate_pipeline(model_path=Path("non_existent_model_file.joblib"))


def test_production_saved_model_artifact_predictions():
    """Verify that the saved production model artifact can load and classify correctly."""
    model_path = Path("models/spam_classifier_pipeline.joblib")
    assert model_path.exists(), "Production model artifact should exist in models/"

    pipe = joblib.load(model_path)

    # Obvious Spam message
    spam_msg = "WINNER!! As a valued customer you have won £1000 cash prize or a free holiday! Call 09061701461 to claim now!"
    spam_pred = pipe.predict([spam_msg])[0]
    spam_proba = pipe.predict_proba([spam_msg])[0]
    assert spam_pred == 1
    assert spam_proba[1] > 0.5

    # Obvious Ham message
    ham_msg = "Hey Mum, I'll be home late today because of traffic. Don't wait for dinner."
    ham_pred = pipe.predict([ham_msg])[0]
    ham_proba = pipe.predict_proba([ham_msg])[0]
    assert ham_pred == 0
    assert ham_proba[0] > 0.5
