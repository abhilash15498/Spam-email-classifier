"""Text preprocessing module for the Spam Message Classifier.

Design Philosophy:
- Preserves high-value spam signals: URLs, phone numbers, and currency indicators
  are replaced with semantic tokens ('url_token', 'phone_token', 'currency_token')
  instead of being discarded.
- Removes noisy formatting: HTML tags, email addresses, and excessive whitespace.
- Normalizes text: Lowercases, tokenizes words, filters standard stop words,
  and applies lemmatization using NLTK's WordNetLemmatizer.
- Robust edge handling: Gracefully handles None, empty strings, numbers, and symbols.
"""

import re
from typing import Optional, Set
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Ensure required NLTK resources are available
def ensure_nltk_resources() -> None:
    """Download required NLTK datasets if not already present."""
    required = ["stopwords", "punkt", "punkt_tab", "wordnet"]
    for resource in required:
        try:
            if resource == "stopwords":
                nltk.data.find("corpora/stopwords")
            elif resource == "wordnet":
                nltk.data.find("corpora/wordnet")
            elif resource.startswith("punkt"):
                nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download(resource, quiet=True)


ensure_nltk_resources()

# Initialize Lemmatizer and Stop Words set
_LEMMATIZER = WordNetLemmatizer()

try:
    _STOP_WORDS: Set[str] = set(stopwords.words("english"))
except Exception:
    nltk.download("stopwords", quiet=True)
    _STOP_WORDS = set(stopwords.words("english"))

# Preserve important negation words that can change semantic meaning
_NEGATIONS = {"no", "not", "nor", "neither", "never", "none"}
_FILTERED_STOP_WORDS = _STOP_WORDS - _NEGATIONS

# Pre-compiled regular expressions for speed and consistency
_RE_HTML = re.compile(r"<[^>]+>")
_RE_URL = re.compile(
    r"(?:https?://\S+|www\.\S+|[a-zA-Z0-9.\-]+\.(?:com|org|net|co|uk|in|info|biz)\b\S*)",
    re.IGNORECASE,
)
_RE_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
_RE_PHONE = re.compile(r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}|\b\d{5,12}\b")
_RE_CURRENCY = re.compile(r"[\$£€¥₹]")
_RE_SPECIAL_CHARS = re.compile(r"[^a-zA-Z0-9\s]")
_RE_WHITESPACE = re.compile(r"\s+")


def clean_text(text: Optional[str], remove_stopwords: bool = True, lemmatize: bool = True) -> str:
    """Preprocess and normalize an input text message for classification.

    Steps:
    1. Handle null/empty/non-string values safely.
    2. Remove HTML tags.
    3. Normalize URLs to 'url_token'.
    4. Normalize email addresses to 'email_token'.
    5. Normalize currency symbols to 'currency_token'.
    6. Normalize phone numbers and long digit sequences to 'phone_token'.
    7. Convert to lowercase.
    8. Remove residual punctuation and non-alphanumeric characters.
    9. Tokenize into words.
    10. Optionally filter stop words (preserving negations).
    11. Optionally lemmatize words.
    12. Reconstruct cleaned text with single spaces.

    Args:
        text: Raw input message string or None.
        remove_stopwords: Whether to remove common English stopwords.
        lemmatize: Whether to reduce words to their dictionary lemmas.

    Returns:
        Cleaned, normalized text string.
    """
    if text is None:
        return ""

    if not isinstance(text, str):
        text = str(text)

    text = text.strip()
    if not text:
        return ""

    # 1. Strip HTML tags
    text = _RE_HTML.sub(" ", text)

    # 2. Normalize Email addresses first (so email domains aren't partially matched as URLs)
    text = _RE_EMAIL.sub(" email_token ", text)

    # 3. Normalize URLs
    text = _RE_URL.sub(" url_token ", text)

    # 4. Normalize Currency symbols
    text = _RE_CURRENCY.sub(" currency_token ", text)

    # 5. Normalize Phone numbers and long numeric codes
    text = _RE_PHONE.sub(" phone_token ", text)

    # 6. Lowercase
    text = text.lower()

    # 7. Remove non-alphanumeric special characters (keeping underscores for tokens)
    text = re.sub(r"[^a-z0-9_\s]", " ", text)

    # 8. Tokenize into words
    tokens = text.split()

    cleaned_tokens = []
    for token in tokens:
        # Ignore empty tokens
        if not token:
            continue

        # Keep special tokens as-is
        if token in {"url_token", "email_token", "currency_token", "phone_token"}:
            cleaned_tokens.append(token)
            continue

        # Stop word filtering
        if remove_stopwords and token in _FILTERED_STOP_WORDS:
            continue

        # Lemmatization
        if lemmatize:
            token = _LEMMATIZER.lemmatize(token)

        if token:
            cleaned_tokens.append(token)

    return " ".join(cleaned_tokens)
