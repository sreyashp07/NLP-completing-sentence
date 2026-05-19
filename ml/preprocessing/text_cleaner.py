"""
Production NLP text preprocessing pipeline.

Handles the full cleaning lifecycle:
- URL/email removal
- Emoji handling
- Contraction expansion
- Lowercasing
- Punctuation normalization
- Stopword removal (optional)
- Lemmatization

Design: Stateless functions + configurable TextCleaner class.
"""
import re
import string
from typing import List, Optional

import nltk
from loguru import logger

# Download required NLTK data (idempotent)
def _download_nltk_resources() -> None:
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            logger.info(f"Downloading NLTK resource: {name}")
            nltk.download(name, quiet=True)


_download_nltk_resources()

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Common English contractions
CONTRACTIONS: dict = {
    "won't": "will not", "can't": "cannot", "n't": " not",
    "'re": " are", "'s": " is", "'d": " would", "'ll": " will",
    "'t": " not", "'ve": " have", "'m": " am",
    "i'm": "i am", "you're": "you are", "he's": "he is",
    "she's": "she is", "it's": "it is", "we're": "we are",
    "they're": "they are", "i've": "i have", "you've": "you have",
    "we've": "we have", "they've": "they have", "i'd": "i would",
    "you'd": "you would", "he'd": "he would", "she'd": "she would",
    "i'll": "i will", "you'll": "you will", "he'll": "he will",
    "she'll": "she will", "we'll": "we will", "they'll": "they will",
    "isn't": "is not", "aren't": "are not", "wasn't": "was not",
    "weren't": "were not", "hasn't": "has not", "haven't": "have not",
    "hadn't": "had not", "doesn't": "does not", "don't": "do not",
    "didn't": "did not", "couldn't": "could not", "wouldn't": "would not",
    "shouldn't": "should not", "mightn't": "might not", "mustn't": "must not",
}

# Emoji to text mapping (common support-related emojis)
EMOJI_MAP: dict = {
    "😡": " angry ", "😤": " frustrated ", "😭": " sad ", "🙏": " please ",
    "❌": " error ", "✅": " success ", "⚠️": " warning ", "🚨": " urgent ",
    "💳": " payment ", "💰": " money ", "🔒": " locked ", "📦": " package ",
    "🐛": " bug ", "⭐": " star ", "👍": " good ", "👎": " bad ",
    "😊": " happy ", "😞": " disappointed ",
}

STOP_WORDS = set(stopwords.words("english"))
# Keep negations — they carry meaning in support messages
NEGATION_WORDS = {"not", "no", "never", "nor", "neither", "nothing", "nowhere"}
FILTERED_STOP_WORDS = STOP_WORDS - NEGATION_WORDS

_lemmatizer = WordNetLemmatizer()


def expand_contractions(text: str) -> str:
    """Expand contractions for better tokenization."""
    text = text.lower()
    for contraction, expansion in CONTRACTIONS.items():
        text = text.replace(contraction, expansion)
    return text


def handle_emojis(text: str, replace: bool = True) -> str:
    """Replace emojis with text descriptions or remove them."""
    if replace:
        for emoji, description in EMOJI_MAP.items():
            text = text.replace(emoji, description)
    # Remove any remaining emojis not in our map
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F9FF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub(" ", text)


def remove_urls(text: str) -> str:
    """Remove HTTP/HTTPS URLs and www links."""
    url_pattern = re.compile(
        r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
        r"|www\.[a-zA-Z0-9]+\.[a-zA-Z]{2,}"
    )
    return url_pattern.sub(" ", text)


def remove_emails(text: str) -> str:
    """Remove email addresses."""
    email_pattern = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
    return email_pattern.sub(" ", text)


def remove_special_chars(text: str, keep_alphanumeric: bool = True) -> str:
    """Remove special characters, optionally keeping alphanumeric only."""
    if keep_alphanumeric:
        # Keep letters, numbers, spaces, and basic punctuation
        text = re.sub(r"[^a-zA-Z0-9\s\.\,\!\?]", " ", text)
    else:
        text = re.sub(r"[^a-zA-Z\s]", " ", text)
    return text


def normalize_whitespace(text: str) -> str:
    """Collapse multiple whitespace into single space."""
    return re.sub(r"\s+", " ", text).strip()


def lemmatize_tokens(tokens: List[str]) -> List[str]:
    """Lemmatize a list of tokens."""
    return [_lemmatizer.lemmatize(token) for token in tokens]


class TextCleaner:
    """
    Configurable text cleaning pipeline for NLP preprocessing.

    Usage:
        cleaner = TextCleaner()
        clean_text = cleaner.clean("My payment failed but money was deducted!!")
    """

    def __init__(
        self,
        remove_urls: bool = True,
        remove_emails: bool = True,
        handle_emojis: bool = True,
        expand_contractions: bool = True,
        lowercase: bool = True,
        remove_special_chars: bool = True,
        remove_stopwords: bool = False,
        lemmatize: bool = True,
        min_token_length: int = 2,
    ):
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.handle_emojis = handle_emojis
        self.expand_contractions = expand_contractions
        self.lowercase = lowercase
        self.remove_special_chars = remove_special_chars
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.min_token_length = min_token_length

    def clean(self, text: str) -> str:
        """
        Run full cleaning pipeline on a single text.

        Args:
            text: Raw customer support message

        Returns:
            Cleaned, normalized text string
        """
        if not text or not isinstance(text, str):
            return ""

        # Step 1: Handle emojis first (before lowercasing)
        if self.handle_emojis:
            text = handle_emojis(text)

        # Step 2: Lowercase
        if self.lowercase:
            text = text.lower()

        # Step 3: Expand contractions
        if self.expand_contractions:
            text = expand_contractions(text)

        # Step 4: Remove URLs and emails
        if self.remove_urls:
            text = remove_urls(text)
        if self.remove_emails:
            text = remove_emails(text)

        # Step 5: Remove special characters
        if self.remove_special_chars:
            text = remove_special_chars(text)

        # Step 6: Tokenize
        tokens = word_tokenize(text)

        # Step 7: Filter short tokens
        tokens = [t for t in tokens if len(t) >= self.min_token_length]

        # Step 8: Remove stopwords
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in FILTERED_STOP_WORDS]

        # Step 9: Lemmatize
        if self.lemmatize:
            tokens = lemmatize_tokens(tokens)

        # Step 10: Rejoin and normalize
        text = " ".join(tokens)
        text = normalize_whitespace(text)

        return text

    def clean_batch(self, texts: List[str]) -> List[str]:
        """Clean a list of texts."""
        return [self.clean(text) for text in texts]


def extract_keywords(text: str, top_n: int = 5) -> List[str]:
    """
    Extract important keywords from text.
    Simple frequency-based extraction after removing stopwords.

    Args:
        text: Input text
        top_n: Number of keywords to return

    Returns:
        List of important keywords
    """
    cleaned = text.lower()
    cleaned = remove_urls(cleaned)
    cleaned = remove_emails(cleaned)
    cleaned = remove_special_chars(cleaned, keep_alphanumeric=False)

    tokens = word_tokenize(cleaned)
    tokens = [t for t in tokens if len(t) > 2 and t not in STOP_WORDS]
    tokens = lemmatize_tokens(tokens)

    # Frequency count
    freq: dict = {}
    for token in tokens:
        freq[token] = freq.get(token, 0) + 1

    # Sort by frequency
    sorted_tokens = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [token for token, _ in sorted_tokens[:top_n]]
