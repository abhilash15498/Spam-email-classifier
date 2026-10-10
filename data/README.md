# SMS Spam Collection Dataset

This folder manages raw and processed data for the **Spam Message Classifier** project.

## Official Source
- **Archive:** UCI Machine Learning Repository
- **Dataset Title:** SMS Spam Collection
- **Original Authors:** Tiago A. Almeida (Federal University of Sao Carlos) & José María Gómez Hidalgo (R&D Department, Optenet)
- **Direct Download URL:** [https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip](https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip)
- **DOI / UCI Record:** [https://doi.org/10.24432/C5CC84](https://doi.org/10.24432/C5CC84)

## Dataset Properties
- **Total records:** 5,574 labeled SMS messages
- **Columns in raw file:** 2 columns (tab-separated, without header)
  1. `label`: `ham` (legitimate) or `spam` (unsolicited / malicious)
  2. `message`: raw text of the SMS message
- **Class distribution:**
  - `ham`: 4,827 messages (86.60%)
  - `spam`: 747 messages (13.40%)
- **Class imbalance ratio:** approximately 6.46 legitimate messages for every 1 spam message
- **Missing values:** 0
- **Duplicate messages:** 403 exact duplicate messages in the original corpus (e.g. standard notifications, common spam blasts)

## Directory Structure
```text
data/
├── README.md               # This documentation file
├── raw/
│   ├── SMSSpamCollection   # Verified raw TSV file from UCI (unmodified)
│   ├── readme              # Official UCI readme & attribution
│   └── .gitkeep
└── processed/
    ├── sms_spam_cleaned.csv# Cleaned and processed dataset ready for modeling
    └── .gitkeep
```

## How to Acquire the Dataset

### Automated Method (Recommended)
Run the dataset loading module from the project root:
```bash
python -m src.spam_classifier.data_loader
```
This automatically verifies local availability or downloads directly from the official UCI archive, validates file structure, and checks data integrity.

### Manual Download Instructions
If direct internet access is restricted:
1. Download `sms+spam+collection.zip` manually from the UCI URL above.
2. Unzip the archive.
3. Place `SMSSpamCollection` directly into `data/raw/SMSSpamCollection`.
4. Verify by running `python -m src.spam_classifier.data_loader`.

## Data Governance & Reproducibility
- The raw file in `data/raw/` is treated as strictly immutable and read-only.
- All cleaning, exploration, and feature extraction operations are performed programmatically to ensure full reproducibility.
