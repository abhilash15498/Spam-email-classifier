"""Dataset acquisition, validation, and loading utilities for SMS Spam Collection.

Dataset Source:
    UCI Machine Learning Repository: SMS Spam Collection
    URL: https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip
    Original creators: Tiago A. Almeida, José María Gómez Hidalgo
"""

import io
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd

# Default project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RAW_DATA_FILE = RAW_DATA_DIR / "SMSSpamCollection"

UCI_DATASET_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"


def download_dataset(
    target_dir: Path = RAW_DATA_DIR,
    url: str = UCI_DATASET_URL,
    force: bool = False,
) -> Path:
    """Download and extract the SMS Spam Collection dataset from UCI repository.

    Args:
        target_dir: Directory where the raw files should be saved.
        url: Direct zip download URL from UCI.
        force: If True, re-download even if the raw file already exists.

    Returns:
        Path to the extracted SMSSpamCollection file.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    raw_file = target_dir / "SMSSpamCollection"

    if raw_file.exists() and not force:
        return raw_file

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "SpamClassifier-Internship-Project/1.0"},
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        zip_content = response.read()

    with zipfile.ZipFile(io.BytesIO(zip_content)) as z:
        for member in z.namelist():
            # Extract only expected files safely avoiding zip slips
            filename = Path(member).name
            if filename in ("SMSSpamCollection", "readme"):
                target_path = target_dir / filename
                with z.open(member) as src, open(target_path, "wb") as dst:
                    dst.write(src.read())

    if not raw_file.exists():
        raise FileNotFoundError(
            f"Expected file 'SMSSpamCollection' was not found in downloaded zip from {url}."
        )

    return raw_file


def load_raw_data(file_path: Optional[Path] = None) -> pd.DataFrame:
    """Load and validate the raw SMS Spam Collection dataset.

    Args:
        file_path: Path to SMSSpamCollection. If None, uses default RAW_DATA_FILE.

    Returns:
        DataFrame with columns ['label', 'message', 'label_num'].

    Raises:
        FileNotFoundError: If the data file does not exist.
        ValueError: If data structure or labels fail validation.
    """
    path = Path(file_path) if file_path is not None else RAW_DATA_FILE

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run data_loader.download_dataset() "
            "or place SMSSpamCollection into data/raw/."
        )

    # Raw file is tab-separated with no headers: col 0 = label, col 1 = text
    # quoting=3 (csv.QUOTE_NONE) ensures quote characters inside SMS are read verbatim
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["label", "message"],
        quoting=3,
        encoding="utf-8",
        dtype=str,
    )

    # Basic structural validation
    if df.shape[1] < 2:
        raise ValueError(f"Expected at least 2 columns, but found {df.shape[1]}")

    # Strip labels and normalize casing
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["message"] = df["message"].fillna("").astype(str)

    # Validate label values
    valid_labels = {"ham", "spam"}
    present_labels = set(df["label"].unique())
    if not present_labels.issubset(valid_labels):
        invalid = present_labels - valid_labels
        raise ValueError(f"Encountered unexpected label(s): {invalid}. Expected only 'ham' or 'spam'.")

    # Map labels to numeric representation (0 = ham, 1 = spam)
    df["label_num"] = df["label"].map({"ham": 0, "spam": 1})

    return df


def validate_and_summarize(df: pd.DataFrame) -> dict:
    """Summarize dataset properties including class distribution and duplicates.

    Args:
        df: Loaded DataFrame.

    Returns:
        Dictionary of summary statistics.
    """
    total_messages = len(df)
    missing_messages = (df["message"].str.strip() == "").sum()
    duplicate_messages = df.duplicated(subset=["message"]).sum()
    class_counts = df["label"].value_counts().to_dict()
    spam_percentage = (class_counts.get("spam", 0) / total_messages * 100) if total_messages else 0.0

    return {
        "total_messages": total_messages,
        "class_counts": class_counts,
        "spam_percentage": round(spam_percentage, 2),
        "missing_messages": int(missing_messages),
        "duplicate_messages": int(duplicate_messages),
    }


if __name__ == "__main__":
    print("Checking / Downloading dataset from UCI...")
    raw_path = download_dataset()
    print(f"Dataset ready at: {raw_path}")
    data = load_raw_data(raw_path)
    summary = validate_and_summarize(data)
    print("\nDataset Validation Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
