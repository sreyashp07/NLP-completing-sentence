"""
Custom stopwords configuration.

Extends NLTK stopwords with domain-specific terms
while preserving negations and important support keywords.
"""
from nltk.corpus import stopwords
import nltk

try:
    NLTK_STOPWORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    NLTK_STOPWORDS = set(stopwords.words("english"))

# Words to always keep — they carry meaning in support context
KEEP_WORDS = {
    "not", "no", "never", "nor", "neither",
    "failed", "broken", "wrong", "error",
    "cant", "cannot", "wont", "dont",
    "urgent", "help", "please", "asap",
}

# Domain-specific words to remove (too common in support context)
DOMAIN_STOPWORDS = {
    "hello", "hi", "hey", "dear", "thanks",
    "thank", "regards", "sincerely", "team",
}

FINAL_STOPWORDS = (NLTK_STOPWORDS - KEEP_WORDS) | DOMAIN_STOPWORDS
