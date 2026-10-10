# 5-Minute Internship Demonstration Guide: SMS Spam Classifier

**Project:** SMS Spam Message Classifier  
**Candidate Name:** Machine Learning & NLP Intern  
**Tech Stack:** Python, Scikit-learn, NLTK, Pandas, Streamlit  

---

## ⏱️ Presentation Timing Breakdown

| Section | Duration | Key Talking Points |
| :--- | :--- | :--- |
| **1. Problem & Context** | 45 sec | Why SMS spam detection matters; cost of False Positives vs False Negatives. |
| **2. Dataset & Preprocessing** | 60 sec | UCI corpus (5,574 SMS, 13.4% spam); why signal-preserving preprocessing beats naive cleaning. |
| **3. Model Pipeline & Comparison** | 60 sec | End-to-end Pipeline; TF-IDF unigram+bigram; Logistic Regression vs Multinomial Naive Bayes. |
| **4. Verified Results & Confusion Matrix** | 45 sec | 99.10% accuracy, 98.60% spam precision, only 2 false positives on held-out test data. |
| **5. Live Application Demo** | 60 sec | Real-time classification of sample ham and spam messages in Streamlit UI. |
| **6. Limitations & Q&A Defense** | 30 sec | Model limitations (adversarial typos, out-of-domain chat) and defense answers. |

---

## 🎤 Slide-by-Slide Talking Script

### Slide 1: Introduction & Problem Statement (0:00 - 0:45)
> "Good morning/afternoon everyone. Today, I'm presenting an internship-ready end-to-end SMS Spam Message Classifier built with Python, Scikit-learn, NLTK, and Streamlit.
>
> In mobile communications, spam isn't just an annoyance—it's an entry point for phishing, identity theft, and financial fraud. 
> However, building an ML classifier for SMS has a critical operational asymmetry:
> - A **False Negative** means an annoying spam message reaches the inbox.
> - A **False Positive** means an urgent medical appointment confirmation, family emergency, or banking OTP is erroneously diverted to the spam folder.
> 
> Therefore, while overall accuracy is important, our primary engineering objective is **maximizing spam precision** while maintaining high recall."

---

### Slide 2: Dataset Characteristics & Signal-Preserving Preprocessing (0:45 - 1:45)
> "We utilized the benchmark UCI SMS Spam Collection dataset, containing 5,574 labeled messages with a natural class imbalance (86.6% ham, 13.4% spam).
>
> During Exploratory Data Analysis, we discovered two vital statistical distinctions:
> 1. **Message Length:** Spam messages average 139 characters (median 149), compared to 71 characters for legitimate messages.
> 2. **Digit Frequency:** Legitimate messages contain an average of only 0.3 digits, whereas spam messages contain an average of **15.8 digits**—due to claim numbers, premium hotlines, and prize amounts.
>
> **Our Key NLP Decision:**
> Rather than naively stripping all punctuation and numbers, our preprocessing module preserves semantic signals:
> - Web links are normalized to `url_token`.
> - Phone numbers and long codes become `phone_token`.
> - Currency signs (£, $, €) become `currency_token`.
> - Critical negations ('not', 'no') are preserved during stop-word filtering.
> - Words are lemmatized to their dictionary base forms using NLTK's WordNetLemmatizer."

---

### Slide 3: Model Architecture & Comparison (1:45 - 2:45)
> "To prevent data leakage, we performed an 80/20 stratified train/test split before fitting our feature extractor.
>
> We encapsulated feature extraction and classification into a scikit-learn `Pipeline`:
> - **TF-IDF Vectorizer:** Word unigrams and bigrams (`ngram_range=(1, 2)`), sublinear term frequency scaling, and vocabulary filtering (`min_df=2`, `max_df=0.95`).
> - We trained two distinct models on the training set:
>   1. **Logistic Regression** (L2 regularized, solver='liblinear')
>   2. **Multinomial Naive Bayes** (Laplace smoothed, alpha=0.1)
>
> When evaluated on the same held-out test split (1,115 messages):
> - Logistic Regression achieved **99.01% Accuracy** and **99.29% Spam Precision**.
> - Multinomial Naive Bayes achieved **99.10% Accuracy**, **98.60% Spam Precision**, and higher **Spam Recall (94.63%)**, giving the top Spam F1-Score of **96.58%**."

---

### Slide 4: Evaluation Metrics & Confusion Matrix (2:45 - 3:30)
> "Looking at our test confusion matrix:
> - **True Negatives:** 964 legitimate messages correctly delivered.
> - **False Positives:** Only **2** legitimate messages out of 966 were misclassified.
> - **False Negatives:** 8 spam messages slipped through.
> - **True Positives:** 141 spam messages intercepted.
>
> This demonstrates an exceptional 99.79% recall on legitimate messages and an extremely low Type I error rate."

---

### Slide 5: Live Application Demonstration (3:30 - 4:30)
> *(Switch to the Streamlit UI)*
> 
> "Here is our interactive Streamlit application.
> 1. First, let's test a conversational message:
>    *Input:* `'Hey Mum, I will be home late today because of traffic. Don't wait for dinner.'*
>    *Result:* **Not Spam (Ham)** with 99.8% confidence.
>
> 2. Next, let's test an aggressive prize scam:
>    *Input:* `'WINNER!! As a valued customer you have won £1000 cash prize or a free holiday! Call 09061701461 to claim now!'*
>    *Result:* **Spam Detected** with 99.9% probability.
>
> 3. We also provide a Batch CSV upload tab allowing enterprise users to classify hundreds of messages at once and download annotated CSV results."

---

### Slide 6: Limitations & Engineering Reflection (4:30 - 5:00)
> "Every real-world ML model has limitations:
> 1. **Adversarial evasion:** Spammers using intentional character substitutions (e.g. `w!n c@sh`) or images can evade n-gram tokenizers.
> 2. **Short modern slang:** Social messaging abbreviations not present in the training corpus can degrade confidence.
> 3. **Future improvements:** Incorporating subword character n-grams or lightweight transformer embeddings (DistilBERT) for semantic awareness under adversarial noise.
>
> Thank you, and I look forward to your questions!"

---

## 🎯 Common Examiner Questions & Winning Answers

### Q1: Why did you choose TF-IDF + Naive Bayes / Logistic Regression instead of BERT or an LLM?
**Answer:**
> "For SMS classification, sub-millisecond latency and ultra-low compute overhead are critical. A TF-IDF pipeline loads in under 20 milliseconds, requires no GPU, has a memory footprint under 5 MB, and delivers 99.10% accuracy with 98.6% precision. For short text messages, the inductive bias of word frequencies and n-grams provides strong signal without the risk of overfitting or heavy deployment infrastructure associated with large transformer models."

### Q2: Why is data leakage such a big deal in NLP, and how did you prevent it?
**Answer:**
> "Data leakage occurs if the TF-IDF vocabulary or IDF weights are computed across the entire dataset prior to splitting. This allows test set word frequencies to leak into training representations, producing artificially inflated evaluation metrics. We strictly split the dataset first using stratified sampling, and fit the vectorizer exclusively on the training partition inside a Scikit-learn Pipeline."

### Q3: Why didn't you use SMOTE to balance the dataset?
**Answer:**
> "SMOTE generates synthetic examples in continuous Euclidean feature space. In high-dimensional, sparse text matrices (TF-IDF), interpolating between sparse vectors creates artificial feature combinations that rarely correspond to valid linguistic patterns. Instead, we handled class imbalance using stratified train-test splits, evaluated class-specific precision/recall and F1-score rather than accuracy alone, and adjusted classification thresholds."
