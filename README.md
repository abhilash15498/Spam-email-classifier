# 🛡️ SMS Spam Message Classifier

An internship-ready Natural Language Processing (NLP) application that classifies SMS messages and short text as **Spam** (unwanted, promotional, or deceptive) or **Not Spam (Ham)** (legitimate messages) using Scikit-learn, NLTK, and Streamlit.

---

## 📌 Project Overview & Objectives

In mobile telecommunications, SMS spam presents severe user security risks including phishing attacks, financial deception, and malware distribution. However, building an ML filter involves an operational trade-off:
- **False Negative (Missed Spam):** An unwanted promotional message lands in the inbox.
- **False Positive (Legitimate Blocked):** An urgent personal message, flight update, or banking OTP is erroneously flagged as spam.

This project implements the complete, production-grade Machine Learning workflow:
1. **Data Acquisition & Integrity:** Ingestion, validation, and provenance tracking for the benchmark UCI SMS Spam Collection.
2. **Exploratory Data Analysis (EDA):** Statistical characterization of message length, word count, digit frequency, and class distribution.
3. **Signal-Preserving Preprocessing:** Normalizing URLs, phone numbers, and currency signs into semantic tokens rather than discarding them.
4. **Leak-Free ML Pipeline:** Scikit-learn `Pipeline` pairing unigram/bigram TF-IDF vectorization with regularized classifiers.
5. **Model Evaluation & Comparison:** Benchmarking Logistic Regression and Multinomial Naive Bayes against a held-out test set with confusion matrix analysis.
6. **Interactive Web Interface:** Streamlit application providing real-time message inference, confidence meters, and CSV batch processing.

---

## 🛠️ Technology Stack

| Layer | Tools |
| :--- | :--- |
| **Language** | Python 3.10+ |
| **Data Handling** | Pandas, NumPy |
| **NLP & Preprocessing** | NLTK (Tokenization, WordNet Lemmatization, Stop Words) |
| **Machine Learning** | Scikit-learn (TF-IDF Vectorizer, Logistic Regression, MultinomialNB, Pipelines) |
| **Visualizations** | Matplotlib, Seaborn |
| **Model Persistence** | Joblib |
| **User Interface** | Streamlit |
| **Testing** | Pytest |

---

## 📂 Repository Structure

```text
spam-message-classifier/
├── app.py                             # Interactive Streamlit application
├── requirements.txt                   # Core project dependencies
├── README.md                          # Comprehensive project documentation
├── .gitignore                         # Git exclusion rules for virtualenvs & artifacts
├── data/
│   ├── README.md                      # Dataset provenance, schema, and manual download steps
│   ├── raw/
│   │   ├── SMSSpamCollection          # Verified raw dataset from UCI (immutable)
│   │   └── readme                     # Official UCI dataset attribution
│   └── processed/
│       ├── sms_spam_cleaned.csv       # Preprocessed messages and extracted length statistics
│       └── test_split.csv             # Fixed held-out test split (20%) for evaluation
├── models/
│   ├── spam_classifier_pipeline.joblib# Serialized winning model pipeline artifact
│   └── model_metadata.json            # Model parameters, training timestamp, and metrics
├── notebooks/                         # Space for exploratory notebooks
├── reports/
│   ├── class_distribution.png         # Bar plot of dataset class proportions
│   ├── message_length_distribution.png# Character and word length comparisons by class
│   ├── confusion_matrix.png           # Annotated confusion matrix heatmap
│   ├── eda_summary.json               # Exploratory data statistics
│   ├── model_comparison.csv           # Performance comparison across candidate models
│   ├── evaluation_metrics.json        # Measured test metrics on held-out test partition
│   └── internship_presentation_guide.md# 5-minute presentation script & examiner defense guide
├── src/
│   └── spam_classifier/
│       ├── __init__.py                # Package initialization
│       ├── data_loader.py             # Dataset downloading, loading, and validation
│       ├── explore.py                 # EDA workflow and report generation
│       ├── preprocessing.py           # Text cleaning, token normalization, lemmatization
│       ├── train.py                   # Model training, comparison, and persistence pipeline
│       └── evaluate.py                # Test set evaluation and confusion matrix generation
└── tests/
    ├── test_data_loader.py            # Unit tests for data loading and schema validation
    ├── test_preprocessing.py          # Unit tests for text cleaning and edge cases
    └── test_model.py                  # Unit tests for training, serialization, and predictions
```

---

## 💾 Dataset Acquisition & Validation

The project uses the **SMS Spam Collection v.1** from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/228/sms+spam+collection):
- **Total records:** 5,574 labeled messages (tab-separated: label, message).
- **Legitimate (Ham):** 4,827 (86.60%)
- **Spam:** 747 (13.40%)
- **Class Imbalance:** ~6.46 legitimate messages per spam message.

### Automated Download
To automatically download, verify, and store the dataset in `data/raw/`:
```bash
python -m src.spam_classifier.data_loader
```

### Manual Download (If offline)
1. Download `sms+spam+collection.zip` from: `https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip`
2. Extract the file `SMSSpamCollection` directly into `data/raw/SMSSpamCollection`.
3. Verify integrity with `python -m src.spam_classifier.data_loader`.

---

## 🚀 Quickstart & Environment Setup

### 1. Clone the Repository
```bash
git clone <repository-url>
cd spam-message-classifier
```

### 2. Set Up a Virtual Environment

**Windows PowerShell:**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Git Bash / macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🏃 Execution Commands

### Step 1: Run Exploratory Data Analysis (EDA)
Computes length metrics and generates visual charts in `reports/`:
```bash
python -m src.spam_classifier.explore
```

### Step 2: Train Model & Generate Pipeline
Splits data (80/20 stratified), fits TF-IDF and candidate models, evaluates both, and saves the top pipeline to `models/`:
```bash
python -m src.spam_classifier.train
```

### Step 3: Evaluate on Held-Out Test Set
Loads the saved pipeline and evaluates performance against `data/processed/test_split.csv`:
```bash
python -m src.spam_classifier.evaluate
```

### Step 4: Run the Streamlit Web Application
Launches the interactive dashboard locally:
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## 🧪 Automated Testing

Run the full Pytest suite:
```bash
python -m pytest -v
```
All 17 automated tests verify dataset loading, edge-case text preprocessing, pipeline serialization, and prediction outputs.

---

## 🔬 NLP Preprocessing & Feature Extraction

### 1. Signal-Preserving Preprocessing (`preprocessing.py`)
Traditional NLP pipelines often strip all numbers and special symbols. In SMS spam classification, this destroys valuable predictive signals:
- **Email addresses** $\rightarrow$ Normalized to `email_token`
- **Web links & URLs** $\rightarrow$ Normalized to `url_token`
- **Currency symbols** (`£`, `$`, `€`) $\rightarrow$ Normalized to `currency_token`
- **Phone numbers & long numeric codes** $\rightarrow$ Normalized to `phone_token`
- **Negations** (`not`, `no`, `never`) $\rightarrow$ Preserved during stop-word filtering to maintain semantic sentiment.
- **Lemmatization** $\rightarrow$ Base word reduction via NLTK's `WordNetLemmatizer`.

### 2. TF-IDF Vectorization
Messages are converted into numerical vectors using Scikit-learn's `TfidfVectorizer`:
- **N-gram Range:** Unigrams and Bigrams `(1, 2)` to capture phrase context (e.g. `"cash prize"`, `"claim now"`).
- **Sublinear TF:** Logarithmic term frequency scaling ($1 + \log(TF)$) to dampen high-frequency repetition.
- **Filtering:** Minimum document frequency `min_df=2`, maximum frequency `max_df=0.95`.

---

## 📊 Measured Evaluation Results

Evaluated strictly on the held-out 20% test partition (1,115 SMS messages, random state 42, stratified):

### Model Comparison
| Model | Overall Accuracy | Spam Precision | Spam Recall | Spam F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** (solver='liblinear', C=5.0) | 99.01% | **99.29%** | 93.29% | 0.9619 |
| **Multinomial Naive Bayes** (alpha=0.1) *(Selected)* | **99.10%** | 98.60% | **94.63%** | **0.9658** |

### Confusion Matrix Breakdown (Test Set: 1,115 samples)
| Actual Class \ Predicted Class | Predicted Ham | Predicted Spam | Total |
| :--- | :---: | :---: | :---: |
| **Actual Ham (Legitimate)** | **964** (True Negative) | **2** (False Positive - Type I) | 966 |
| **Actual Spam** | **8** (False Negative - Type II) | **141** (True Positive) | 149 |

- **Legitimate Message Recall:** $964 / 966 = \mathbf{99.79\%}$ (Only 2 legitimate messages misclassified out of 966).
- **Spam Precision:** $141 / (141 + 2) = \mathbf{98.60\%}$.

---

## 🖥️ Streamlit Application Features

- **Single Message Classification:** Real-time inference with probability breakdown meters.
- **Visual Feedback:** Color-coded alert boxes (Green for Legitimate, Red for Spam).
- **One-Click Demo Presets:** Preloaded test messages for rapid live demonstration.
- **Batch CSV Upload:** Upload any CSV with a `message` column, run batch classification, and download results.
- **Architecture Transparency:** Sidebar displaying live model parameters, test metrics, and confusion matrix.

---

## ⚠️ Limitations & Future Work

1. **Adversarial Character Substitution:** Spammers using intentional misspellings (e.g. `Fr33 c@sh`) can occasionally bypass dictionary lemmatizers. Subword/character n-grams or byte-pair encoding (BPE) could improve robustness.
2. **Short Modern Slang:** Gen-Z slang and modern chat abbreviations absent in the training corpus can lower model confidence.
3. **Multilingual SMS:** The dataset is predominantly English. Multi-lingual classification would require multilingual sentence embeddings.

---

## 📜 Attribution & License

- **Dataset Authors:** Tiago A. Almeida and José María Gómez Hidalgo.
- **Dataset Citation:** Almeida, T.A., Gómez Hidalgo, J.M., Yamakami, A. (2011). *Contributions to the Study of SMS Spam Filtering: New Collection and Results*. Proceedings of the 2011 ACM Symposium on Document Engineering (DocEng '11).
- **Repository License:** Provided for educational and research use under open-source terms.
