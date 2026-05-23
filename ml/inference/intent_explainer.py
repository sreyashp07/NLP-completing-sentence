"""
Intent prediction explainability module.

Provides human-readable explanations for why a particular
intent was predicted. Uses TF-IDF feature weights to identify
the most influential words in the prediction.
"""
import pickle
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np


MODEL_DIR = Path("ml/saved_models/baseline")


def get_top_features(
    text: str,
    pipeline,
    label_encoder,
    predicted_class: str,
    top_n: int = 5,
) -> List[Tuple[str, float]]:
    """
    Get top TF-IDF features that influenced the prediction.

    Args:
        text: Input customer message
        pipeline: Trained sklearn pipeline
        label_encoder: Fitted LabelEncoder
        predicted_class: The predicted intent label
        top_n: Number of top features to return

    Returns:
        List of (word, weight) tuples sorted by importance
    """
    vectorizer = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["clf"]

    class_idx = list(label_encoder.classes_).index(predicted_class)
    feature_names = vectorizer.get_feature_names_out()
    class_coefs = classifier.coef_[class_idx]

    text_vector = vectorizer.transform([text])
    text_array = text_vector.toarray()[0]

    weighted = text_array * class_coefs
    top_indices = np.argsort(weighted)[::-1][:top_n]

    result = []
    for idx in top_indices:
        if text_array[idx] > 0:
            result.append((feature_names[idx], float(weighted[idx])))

    return result


def explain_prediction(text: str, prediction: Dict) -> Dict:
    """
    Generate a human-readable explanation for a prediction.

    Args:
        text: Original customer message
        prediction: Prediction result dict from predictor

    Returns:
        Explanation dict with influential words and reasoning
    """
    try:
        with open(MODEL_DIR / "pipeline.pkl", "rb") as f:
            pipeline = pickle.load(f)
        with open(MODEL_DIR / "label_encoder.pkl", "rb") as f:
            le = pickle.load(f)

        intent = prediction["primary_intent"]["intent"]
        confidence = prediction["primary_intent"]["confidence"]
        top_features = get_top_features(text, pipeline, le, intent, top_n=5)

        confidence_label = (
            "very high" if confidence >= 0.9 else
            "high" if confidence >= 0.75 else
            "medium" if confidence >= 0.6 else
            "low"
        )

        return {
            "intent": intent,
            "confidence": confidence,
            "confidence_label": confidence_label,
            "influential_words": top_features,
            "explanation": (
                f"Classified as '{intent}' with {confidence_label} confidence "
                f"({confidence*100:.1f}%). Key signals: "
                f"{', '.join([w for w, _ in top_features[:3]])}."
            ),
        }
    except Exception as e:
        return {"error": str(e), "explanation": "Explanation unavailable"}
