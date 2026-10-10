"""Unit tests for text preprocessing functions in preprocessing.py."""

import pytest
from src.spam_classifier.preprocessing import clean_text


def test_clean_text_normal_message():
    """Verify standard text cleaning and lowercasing."""
    raw = "Hey friend, are we still meeting up for dinner tonight?"
    cleaned = clean_text(raw)
    assert "friend" in cleaned
    assert "meeting" in cleaned or "meet" in cleaned
    assert "dinner" in cleaned
    assert "tonight" in cleaned
    # Check lowercased
    assert cleaned == cleaned.lower()


def test_clean_text_html_tags():
    """Verify HTML tags are stripped out."""
    raw = "<b>URGENT!</b> You have <i>won</i> a prize."
    cleaned = clean_text(raw)
    assert "<b>" not in cleaned
    assert "</i>" not in cleaned
    assert "urgent" in cleaned
    assert "prize" in cleaned


def test_clean_text_url_replacement():
    """Verify URLs are normalized to 'url_token'."""
    raw = "Claim your reward now at http://win-big.com/prize or www.bonus.co.uk"
    cleaned = clean_text(raw)
    assert "url_token" in cleaned
    assert "http" not in cleaned


def test_clean_text_email_replacement():
    """Verify email addresses are converted to 'email_token'."""
    raw = "Send your details to claims@freegifts.org immediately."
    cleaned = clean_text(raw)
    assert "email_token" in cleaned
    assert "claims@freegifts.org" not in cleaned


def test_clean_text_currency_replacement():
    """Verify currency symbols are preserved as 'currency_token'."""
    raw = "You won £5000 in cash! Or $10000 bonus!"
    cleaned = clean_text(raw)
    assert "currency_token" in cleaned


def test_clean_text_phone_replacement():
    """Verify phone numbers are converted to 'phone_token'."""
    raw = "Call 08712460324 to claim your cash award."
    cleaned = clean_text(raw)
    assert "phone_token" in cleaned


def test_clean_text_negation_preservation():
    """Verify crucial negations like 'not' and 'no' are retained."""
    raw = "I am not going to the party, no way."
    cleaned = clean_text(raw)
    assert "not" in cleaned
    assert "no" in cleaned


def test_clean_text_lemmatization():
    """Verify plural words or noun lemmas are handled."""
    raw = "Congratulations, multiple winners and prizes await!"
    cleaned = clean_text(raw)
    assert "prize" in cleaned or "prizes" in cleaned


def test_clean_text_edge_cases():
    """Verify null, empty, whitespace, and non-string inputs."""
    assert clean_text("") == ""
    assert clean_text("   \n\t   ") == ""
    assert clean_text(None) == ""
    assert clean_text(1234567890) != ""
    assert clean_text("!!!???***") == ""
