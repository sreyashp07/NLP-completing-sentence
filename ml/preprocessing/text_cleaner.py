"""
Production NLP text preprocessing pipeline.
Enhanced with better contraction handling and emoji support.
"""
import re
import string
from typing import List, Optional

import nltk

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
            nltk.download(name, quiet=True)

_download_nltk_resources()

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

CONTRACTIONS = {
    "won't": "will not", "can't": "cannot", "n't": " not",
    "'re": " are", "'s": " is", "'d": " would", "'ll": " will",
    "'ve": " have", "'m": " am", "i'm": "i am", "you're": "you are",
    "it's": "it is", "we're": "we are", "they're": "they are",
    "i've": "i have", "you've": "you have", "we've": "we have",
    "i'd": "i would", "you'd": "you would", "i'll": "i will",
    "you'll": "you will", "we'll": "we will", "they'll": "they will",
    "isn't": "is not", "aren't": "are not", "wasn't": "was not",
    "weren't": "were not", "hasn't": "has not", "haven't": "have not",
    "hadn't": "had not", "doesn't": "does not", "don't": "do not",
    "didn't": "did not", "couldn't": "could not", "wouldn't": "would not",
    "shouldn't": "should not", "mightn't": "might not", "mustn't": "must not",
    "needn't": "need not", "daren't": "dare not", "oughtn't": "ought not",
}

EMOJI_MAP = {
    "😡": " angry ", "😤": " frustrated ", "😭": " sad ",
    "🙏": " please ", "❌": " error ", "✅": " success ",
    "⚠️": " warning ", "🚨": " urgent ", "💳": " payment ",
    "💰": " money ", "🔒": " locked ", "📦": " package ",
    "🐛": " bug ", "⭐": " star ", "👍": " good ", "👎": " bad ",
    "😊": " happy ", "😞": " disappointed ", "🔥": " urgent ",
    "💡": " idea ", "📧": " email ", "📱": " mobile ",
}

STOP_WORDS = set(stopwords.words("english"))
NEGATION_WORDS = {"not", "no", "never", "nor", "neither", "nothing", "nowhere"}
FILTERED_STOP_WORDS = STOP_WORDS - NEGATION_WORDS
_lemmatizer = WordNetLemmatizer()


def expand_contractions(text: str) -> str:
    text = text.lower()
    for contraction, expansion in CONTRACTIONS.items():
        text = text.replace(contraction, expansion)
    return text


def handle_emojis(text: str, replace: bool = True) -> str:
    if replace:
        for emoji, description in EMOJI_MAP.items():
            text = text.replace(emoji, description)
    emoji_pattern = re.compile(
        "["u"\U0001F600-\U0001F64F"u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F9FF"u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251""]+", flags=re.UNICODE,
    )
    return emoji_pattern.sub(" ", text)


def remove_urls(text: str) -> str:
    url_pattern = re.compile(
        r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
        r"|www\.[a-zA-Z0-9]+\.[a-zA-Z]{2,}"
    )
    return url_pattern.sub(" ", text)


def remove_emails(text: str) -> str:
    return re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", " ", text)


def remove_special_chars(text: str, keep_alphanumeric: bool = True) -> str:
    if keep_alphanumeric:
        text = re.sub(r"[^a-zA-Z0-9\s\.\,\!\?]", " ", text)
    else:
        text = re.sub(r"[^a-zA-Z\s]", " ", text)
    return text


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def lemmatize_tokens(tokens: List[str]) -> List[str]:
    return [_lemmatizer.lemmatize(token) for token in tokens]


class TextCleaner:
    """
    Configurable text cleaning pipeline for NLP preprocessing.

    Usage:
        cleaner = TextCleaner()
        clean = cleaner.clean("My payment failed!!")
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
        if not text or not isinstance(text, str):
            return ""
        if self.handle_emojis:
            text = handle_emojis(text)
        if self.lowercase:
            text = text.lower()
        if self.expand_contractions:
            text = expand_contractions(text)
        if self.remove_urls:
            text = remove_urls(text)
        if self.remove_emails:
            text = remove_emails(text)
        if self.remove_special_chars:
            text = remove_special_chars(text)
        tokens = word_tokenize(text)
        tokens = [t for t in tokens if len(t) >= self.min_token_length]
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in FILTERED_STOP_WORDS]
        if self.lemmatize:
            tokens = lemmatize_tokens(tokens)
        return normalize_whitespace(" ".join(tokens))

    def clean_batch(self, texts: List[str]) -> List[str]:
        return [self.clean(text) for text in texts]


def extract_keywords(text: str, top_n: int = 5) -> List[str]:
    cleaned = remove_urls(text.lower())
    cleaned = remove_emails(cleaned)
    cleaned = remove_special_chars(cleaned, keep_alphanumeric=False)
    tokens = word_tokenize(cleaned)
    tokens = [t for t in tokens if len(t) > 2 and t not in STOP_WORDS]
    tokens = lemmatize_tokens(tokens)
    freq: dict = {}
    for token in tokens:
        freq[token] = freq.get(token, 0) + 1
    sorted_tokens = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [token for token, _ in sorted_tokens[:top_n]]
