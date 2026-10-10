"""Unit tests for dataset loading and validation functions."""

import tempfile
from pathlib import Path
import pytest
import pandas as pd

from src.spam_classifier.data_loader import load_raw_data, validate_and_summarize


def test_load_raw_data_valid_file():
    """Verify loading and basic integrity of the real or mock dataset."""
    df = load_raw_data()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 5574
    assert list(df.columns) == ["label", "message", "label_num"]
    assert set(df["label"].unique()) == {"ham", "spam"}
    assert set(df["label_num"].unique()) == {0, 1}


def test_load_raw_data_missing_file_raises_error():
    """Verify that attempting to load a non-existent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_raw_data(Path("non_existent_directory/fake_file.tsv"))


def test_load_raw_data_invalid_labels():
    """Verify that unexpected labels trigger a ValueError."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".tsv", encoding="utf-8") as tmp:
        tmp.write("ham\tHello there\nunknown_label\tSuspicious message\n")
        tmp_path = Path(tmp.name)

    try:
        with pytest.raises(ValueError, match="Encountered unexpected label"):
            load_raw_data(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def test_validate_and_summarize():
    """Verify dataset summary statistics calculation."""
    df = pd.DataFrame({
        "label": ["ham", "spam", "ham", "ham"],
        "message": ["Hi", "Win cash now", "Hi", ""],
        "label_num": [0, 1, 0, 0],
    })
    summary = validate_and_summarize(df)
    assert summary["total_messages"] == 4
    assert summary["class_counts"] == {"ham": 3, "spam": 1}
    assert summary["spam_percentage"] == 25.0
    assert summary["missing_messages"] == 1
    assert summary["duplicate_messages"] == 1
