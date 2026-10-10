"""Exploratory Data Analysis (EDA) module for SMS Spam Collection.

Generates reproducible statistical summaries and visual reports under `reports/`.
Saves preprocessed dataset to `data/processed/sms_spam_cleaned.csv`.
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.spam_classifier.data_loader import load_raw_data, validate_and_summarize
from src.spam_classifier.preprocessing import clean_text

# Style settings for clean, professional visualizations
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"font.size": 11, "figure.autolayout": True})

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def perform_eda(save_plots: bool = True) -> pd.DataFrame:
    """Run full exploratory data analysis on the SMS Spam Collection dataset.

    Args:
        save_plots: If True, saves visual charts in reports/ directory.

    Returns:
        DataFrame enriched with text length statistics and cleaned text.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = load_raw_data()
    summary = validate_and_summarize(df)

    # Compute length features for analysis
    df["char_count"] = df["message"].apply(len)
    df["word_count"] = df["message"].apply(lambda s: len(s.split()))
    df["digit_count"] = df["message"].apply(lambda s: sum(c.isdigit() for c in s))

    # Compute preprocessed text
    print("Applying text preprocessing to all messages...")
    df["clean_message"] = df["message"].apply(clean_text)
    df["clean_word_count"] = df["clean_message"].apply(lambda s: len(s.split()))

    # Statistical breakdown by class
    stats_by_class = (
        df.groupby("label")[["char_count", "word_count", "digit_count"]]
        .agg(["mean", "median", "std", "min", "max"])
        .to_dict()
    )

    eda_report = {
        "dataset_summary": summary,
        "class_statistics": {
            "ham": {
                "char_mean": float(df[df["label"] == "ham"]["char_count"].mean()),
                "char_median": float(df[df["label"] == "ham"]["char_count"].median()),
                "word_mean": float(df[df["label"] == "ham"]["word_count"].mean()),
                "word_median": float(df[df["label"] == "ham"]["word_count"].median()),
                "digit_mean": float(df[df["label"] == "ham"]["digit_count"].mean()),
            },
            "spam": {
                "char_mean": float(df[df["label"] == "spam"]["char_count"].mean()),
                "char_median": float(df[df["label"] == "spam"]["char_count"].median()),
                "word_mean": float(df[df["label"] == "spam"]["word_count"].mean()),
                "word_median": float(df[df["label"] == "spam"]["word_count"].median()),
                "digit_mean": float(df[df["label"] == "spam"]["digit_count"].mean()),
            },
        },
    }

    # Save summary report
    summary_path = REPORTS_DIR / "eda_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(eda_report, f, indent=2)
    print(f"Saved EDA summary metrics to: {summary_path}")

    # Save preprocessed dataset
    processed_path = PROCESSED_DIR / "sms_spam_cleaned.csv"
    df.to_csv(processed_path, index=False, encoding="utf-8")
    print(f"Saved preprocessed data to: {processed_path}")

    if save_plots:
        # Plot 1: Class Distribution
        fig, ax = plt.subplots(figsize=(7, 5))
        class_counts = df["label"].value_counts()
        colors = ["#2b5c8f", "#d9534f"]
        bars = ax.bar(["Ham (Legitimate)", "Spam"], class_counts.values, color=colors, width=0.55)
        for bar in bars:
            height = bar.get_height()
            percentage = (height / len(df)) * 100
            ax.annotate(
                f"{height:,} ({percentage:.1f}%)",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontweight="bold",
            )
        ax.set_title("SMS Spam Collection: Class Distribution", fontsize=13, fontweight="bold", pad=12)
        ax.set_ylabel("Number of Messages", fontsize=11)
        ax.set_ylim(0, max(class_counts.values) * 1.15)
        plot_path_1 = REPORTS_DIR / "class_distribution.png"
        fig.savefig(plot_path_1, dpi=300)
        plt.close(fig)
        print(f"Saved plot: {plot_path_1}")

        # Plot 2: Message Length Comparison (Characters and Words)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        # Character length histogram (capped at 99th percentile for clean display)
        p99_char = df["char_count"].quantile(0.99)
        sns.histplot(
            data=df[df["char_count"] <= p99_char],
            x="char_count",
            hue="label",
            bins=40,
            kde=True,
            palette={"ham": "#2b5c8f", "spam": "#d9534f"},
            alpha=0.6,
            ax=ax1,
        )
        ax1.set_title("Distribution of Message Length (Characters)", fontweight="bold")
        ax1.set_xlabel("Character Count")
        ax1.set_ylabel("Frequency")

        # Word count histogram (capped at 99th percentile)
        p99_word = df["word_count"].quantile(0.99)
        sns.histplot(
            data=df[df["word_count"] <= p99_word],
            x="word_count",
            hue="label",
            bins=35,
            kde=True,
            palette={"ham": "#2b5c8f", "spam": "#d9534f"},
            alpha=0.6,
            ax=ax2,
        )
        ax2.set_title("Distribution of Word Count", fontweight="bold")
        ax2.set_xlabel("Word Count")
        ax2.set_ylabel("Frequency")

        plot_path_2 = REPORTS_DIR / "message_length_distribution.png"
        fig.savefig(plot_path_2, dpi=300)
        plt.close(fig)
        print(f"Saved plot: {plot_path_2}")

    return df


if __name__ == "__main__":
    print("Executing Exploratory Data Analysis...")
    df_explored = perform_eda()
    print("EDA complete.")
