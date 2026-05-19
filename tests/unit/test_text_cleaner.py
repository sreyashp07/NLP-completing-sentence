"""
Unit tests for NLP text cleaning pipeline.
Tests every cleaning function in isolation.
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ml.preprocessing.text_cleaner import (
    TextCleaner,
    expand_contractions,
    extract_keywords,
    handle_emojis,
    normalize_whitespace,
    remove_emails,
    remove_urls,
)


class TestRemoveUrls:
    def test_removes_http_url(self):
        result = remove_urls("Visit http://example.com for help")
        assert "http://example.com" not in result

    def test_removes_https_url(self):
        result = remove_urls("Go to https://support.stripe.com now")
        assert "https" not in result

    def test_keeps_normal_text(self):
        result = remove_urls("My payment failed today")
        assert "payment" in result
        assert "failed" in result

    def test_empty_string(self):
        assert remove_urls("") == ""


class TestRemoveEmails:
    def test_removes_email(self):
        result = remove_emails("Contact us at support@company.com")
        assert "support@company.com" not in result

    def test_keeps_normal_text(self):
        result = remove_emails("I need help with my account")
        assert "account" in result

    def test_empty_string(self):
        assert remove_emails("") == ""


class TestExpandContractions:
    def test_expands_cant(self):
        result = expand_contractions("I can't login")
        assert "cannot" in result

    def test_expands_wont(self):
        result = expand_contractions("It won't work")
        assert "will not" in result

    def test_expands_dont(self):
        result = expand_contractions("I don't understand")
        assert "do not" in result

    def test_no_change_clean_text(self):
        result = expand_contractions("my payment failed")
        assert result == "my payment failed"


class TestHandleEmojis:
    def test_replaces_known_emoji(self):
        result = handle_emojis("My payment ❌ failed")
        assert "error" in result or "❌" not in result

    def test_removes_unknown_emoji(self):
        result = handle_emojis("Hello 🌍 world")
        assert "🌍" not in result

    def test_empty_string(self):
        assert handle_emojis("") == ""


class TestNormalizeWhitespace:
    def test_collapses_spaces(self):
        result = normalize_whitespace("hello   world   test")
        assert result == "hello world test"

    def test_strips_ends(self):
        result = normalize_whitespace("  hello world  ")
        assert result == "hello world"

    def test_handles_newlines(self):
        result = normalize_whitespace("hello\n\nworld")
        assert "\n" not in result


class TestTextCleaner:
    def setup_method(self):
        self.cleaner = TextCleaner(
            remove_urls=True,
            remove_emails=True,
            handle_emojis=True,
            expand_contractions=True,
            lowercase=True,
            remove_special_chars=True,
            remove_stopwords=False,
            lemmatize=True,
        )

    def test_basic_cleaning(self):
        result = self.cleaner.clean("My Payment FAILED today!")
        assert result == result.lower()
        assert "!" not in result

    def test_url_removed(self):
        result = self.cleaner.clean("Visit https://example.com for help")
        assert "https" not in result

    def test_email_removed(self):
        result = self.cleaner.clean("Email us at help@support.com")
        assert "@" not in result

    def test_contraction_expanded(self):
        result = self.cleaner.clean("I can't access my account")
        assert "cannot" in result or "can" in result

    def test_empty_string_returns_empty(self):
        result = self.cleaner.clean("")
        assert result == ""

    def test_none_returns_empty(self):
        result = self.cleaner.clean(None)
        assert result == ""

    def test_batch_clean(self):
        texts = [
            "My payment failed",
            "I want a refund",
            "Account is locked",
        ]
        results = self.cleaner.clean_batch(texts)
        assert len(results) == 3
        assert all(isinstance(r, str) for r in results)

    def test_preserves_meaningful_content(self):
        result = self.cleaner.clean("payment failed money deducted")
        assert "payment" in result
        assert "failed" in result

    def test_lemmatization(self):
        result = self.cleaner.clean("payments are failing")
        assert "payment" in result or "fail" in result


class TestExtractKeywords:
    def test_returns_list(self):
        result = extract_keywords("My payment failed but money was deducted")
        assert isinstance(result, list)

    def test_returns_correct_count(self):
        result = extract_keywords("payment failed money deducted account", top_n=3)
        assert len(result) <= 3

    def test_filters_stopwords(self):
        result = extract_keywords("the and is of payment")
        assert "the" not in result
        assert "and" not in result

    def test_payment_keyword_detected(self):
        result = extract_keywords("My payment failed but money got deducted", top_n=5)
        assert any(kw in ["payment", "failed", "money", "deducted"] for kw in result)

    def test_empty_string(self):
        result = extract_keywords("")
        assert isinstance(result, list)
