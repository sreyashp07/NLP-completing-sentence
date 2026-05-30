"""
Unified text processing pipeline.
Chains all preprocessing steps in correct order.
"""
from typing import Dict, List
from ml.preprocessing.text_cleaner import TextCleaner, extract_keywords
from ml.preprocessing.text_normalizer import normalize_text
from ml.preprocessing.sentiment_analyzer import analyze_sentiment
from ml.preprocessing.language_detector import detect_language


class TextPipeline:
    """Full text preprocessing pipeline."""

    def __init__(self):
        self.cleaner = TextCleaner(remove_stopwords=False, lemmatize=True)

    def process(self, text: str) -> Dict:
        if not text or not isinstance(text, str):
            return {"original": text, "cleaned": "", "normalized": "",
                    "sentiment": {}, "language": {}, "keywords": []}

        normalized = normalize_text(text)
        cleaned = self.cleaner.clean(normalized)
        sentiment = analyze_sentiment(text)
        language = detect_language(text)
        keywords = extract_keywords(text, top_n=5)

        return {
            "original": text,
            "normalized": normalized,
            "cleaned": cleaned,
            "sentiment": sentiment,
            "language": language,
            "keywords": keywords,
            "word_count": len(cleaned.split()),
            "char_count": len(cleaned),
        }

    def process_batch(self, texts: List[str]) -> List[Dict]:
        return [self.process(text) for text in texts]
