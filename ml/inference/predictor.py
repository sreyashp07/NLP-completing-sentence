"""
Unified inference engine for intent classification.

Abstracts away which model is being used (baseline vs transformer).
The API layer only interacts with this class — never directly with models.

Design pattern: Strategy Pattern — swap models without changing calling code.
"""
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml
from loguru import logger

from ml.preprocessing.text_cleaner import TextCleaner, extract_keywords

# Intent metadata — maps intent label to department/priority
INTENT_METADATA: Dict[str, Dict] = {
    "payment_issue":      {"department": "Billing Team",    "priority": "critical", "label": "Payment Issue"},
    "refund_request":     {"department": "Billing Team",    "priority": "high",     "label": "Refund Request"},
    "account_locked":     {"department": "Security Team",   "priority": "critical", "label": "Account Locked"},
    "technical_bug":      {"department": "Engineering Team","priority": "high",     "label": "Technical Bug"},
    "feature_request":    {"department": "Product Team",    "priority": "low",      "label": "Feature Request"},
    "subscription_cancel":{"department": "Retention Team",  "priority": "high",     "label": "Subscription Cancel"},
    "invoice_problem":    {"department": "Finance Team",    "priority": "medium",   "label": "Invoice Problem"},
    "shipping_delay":     {"department": "Logistics Team",  "priority": "medium",   "label": "Shipping Delay"},
    "general_inquiry":    {"department": "General Support", "priority": "low",      "label": "General Inquiry"},
}

PRIORITY_ORDER = {"critical": 4, "high": 3, "medium": 2, "low": 1}


class BaselinePredictor:
    """
    Inference wrapper for TF-IDF + Logistic Regression pipeline.
    Loads artifacts from disk and exposes a clean predict() interface.
    """

    def __init__(self, model_dir: str = "ml/saved_models/baseline"):
        self.model_dir = Path(model_dir)
        self.pipeline = None
        self.label_encoder = None
        self.cleaner = TextCleaner(
            remove_stopwords=False,
            lemmatize=True,
        )
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Load model pipeline and label encoder from disk."""
        pipeline_path = self.model_dir / "pipeline.pkl"
        le_path = self.model_dir / "label_encoder.pkl"

        if not pipeline_path.exists():
            raise FileNotFoundError(
                f"No trained model found at {pipeline_path}. "
                "Run `python ml/training/train_baseline.py` first."
            )

        with open(pipeline_path, "rb") as f:
            self.pipeline = pickle.load(f)
        with open(le_path, "rb") as f:
            self.label_encoder = pickle.load(f)

        logger.info(f"Baseline model loaded from {self.model_dir}")
        logger.info(f"Classes: {list(self.label_encoder.classes_)}")

    def predict(self, text: str) -> Dict:
        """
        Run full prediction pipeline on a single text.

        Args:
            text: Raw customer support message

        Returns:
            Prediction dict with intent, confidence, department, priority,
            top_intents, keywords, processing_time_ms
        """
        start = time.time()

        # Preprocess
        cleaned = self.cleaner.clean(text)

        # Get class probabilities
        probs = self.pipeline.predict_proba([cleaned])[0]
        classes = self.label_encoder.classes_

        # Sort by confidence
        ranked = sorted(
            zip(classes, probs),
            key=lambda x: x[1],
            reverse=True,
        )

        primary_intent, primary_conf = ranked[0]
        meta = INTENT_METADATA.get(
            primary_intent,
            {"department": "General Support", "priority": "low", "label": primary_intent}
        )

        # Top-3 candidates
        top_intents = [
            {
                "intent": intent,
                "confidence": float(conf),
                "display_label": INTENT_METADATA.get(intent, {}).get("label", intent),
            }
            for intent, conf in ranked[:3]
        ]

        # Extract keywords for explainability
        keywords = extract_keywords(text, top_n=5)

        elapsed_ms = (time.time() - start) * 1000

        return {
            "text": text,
            "cleaned_text": cleaned,
            "primary_intent": {
                "intent": primary_intent,
                "confidence": float(primary_conf),
                "department": meta["department"],
                "priority": meta["priority"],
                "display_label": meta["label"],
            },
            "top_intents": top_intents,
            "keywords": keywords,
            "model_used": "TF-IDF + Logistic Regression",
            "processing_time_ms": round(elapsed_ms, 2),
        }

    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """Run prediction on a list of texts."""
        return [self.predict(text) for text in texts]

    @property
    def is_loaded(self) -> bool:
        return self.pipeline is not None and self.label_encoder is not None


class IntentRouter:
    """
    High-level router that combines intent prediction with routing logic.

    This is the main entry point for the API layer.
    Supports multiple model backends via strategy pattern.
    """

    def __init__(self, model_type: str = "baseline"):
        self.model_type = model_type
        self.predictor: Optional[BaselinePredictor] = None
        self._initialize()

    def _initialize(self) -> None:
        """Load the appropriate model based on model_type."""
        if self.model_type == "baseline":
            self.predictor = BaselinePredictor()
        else:
            # Transformer model — implement when needed
            logger.warning(f"Model type '{self.model_type}' not yet implemented. Using baseline.")
            self.predictor = BaselinePredictor()

        logger.info(f"IntentRouter initialized with model: {self.model_type}")

    def route(self, text: str) -> Dict:
        """
        Full routing pipeline: classify intent → assign department → set priority.

        Args:
            text: Customer support message

        Returns:
            Complete routing result dict
        """
        if not self.predictor or not self.predictor.is_loaded:
            raise RuntimeError("Model not loaded. Cannot process prediction.")

        result = self.predictor.predict(text)

        # Routing decision log
        intent = result["primary_intent"]["intent"]
        dept = result["primary_intent"]["department"]
        priority = result["primary_intent"]["priority"]
        conf = result["primary_intent"]["confidence"]

        logger.info(
            f"ROUTE | intent={intent} | dept={dept} | priority={priority} | conf={conf:.3f}"
        )

        return result

    def route_batch(self, texts: List[str]) -> List[Dict]:
        """Route a batch of customer messages."""
        return [self.route(text) for text in texts]

    @property
    def is_ready(self) -> bool:
        return self.predictor is not None and self.predictor.is_loaded
