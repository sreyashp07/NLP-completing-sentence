"""
Baseline model training: TF-IDF + Logistic Regression.

This is the production baseline — fast, interpretable, and surprisingly
effective for intent classification. Always build a strong baseline before
jumping to transformers.

Why TF-IDF + LR?
- Trains in seconds (vs hours for BERT)
- Highly interpretable — we can explain every prediction
- Serves as performance benchmark for transformer models
- Can deploy on minimal compute (important for cost)

Run:
    python ml/training/train_baseline.py
"""
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ml.preprocessing.text_cleaner import TextCleaner

# ── Config ─────────────────────────────────────────────────────────────────────

DATA_PATH = "data/raw/customer_support_dataset.csv"
MODEL_DIR = Path("ml/saved_models/baseline")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TFIDF_PARAMS = {
    "max_features": 15000,
    "ngram_range": (1, 3),
    "sublinear_tf": True,          # Apply log(1 + tf) — reduces impact of very common terms
    "min_df": 2,                    # Ignore terms appearing in fewer than 2 docs
    "max_df": 0.95,                 # Ignore terms in more than 95% of docs
    "analyzer": "word",
    "strip_accents": "unicode",
}

LR_PARAMS = {
    "C": 5.0,                       # Regularization — tuned for this dataset size
    "max_iter": 1000,
    "class_weight": "balanced",     # Handle class imbalance automatically
    "solver": "lbfgs",
    "random_state": 42,
}

# ── Training ───────────────────────────────────────────────────────────────────

def load_and_preprocess_data(data_path: str) -> tuple:
    """Load CSV and run NLP preprocessing pipeline."""
    logger.info(f"Loading dataset from {data_path}")
    df = pd.read_csv(data_path)

    logger.info(f"Raw dataset: {len(df)} samples, {df['intent'].nunique()} classes")
    logger.info(f"Class distribution:\n{df['intent'].value_counts()}")

    # Run text cleaning pipeline
    cleaner = TextCleaner(
        remove_urls=True,
        remove_emails=True,
        handle_emojis=True,
        expand_contractions=True,
        lowercase=True,
        remove_special_chars=True,
        remove_stopwords=False,   # Keep stopwords for TF-IDF (they help with context)
        lemmatize=True,
    )

    logger.info("Running NLP preprocessing pipeline...")
    start = time.time()
    df["cleaned_text"] = cleaner.clean_batch(df["text"].tolist())
    elapsed = time.time() - start
    logger.info(f"Preprocessing complete: {elapsed:.2f}s for {len(df)} samples")

    # Encode labels
    le = LabelEncoder()
    df["label"] = le.fit_transform(df["intent"])

    return df, le


def train_model(df: pd.DataFrame) -> tuple:
    """Train TF-IDF + Logistic Regression pipeline."""
    X = df["cleaned_text"].to_numpy()
    y = df["label"].to_numpy()

    # Train/test split with stratification to preserve class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logger.info(f"Train: {len(X_train)} | Test: {len(X_test)}")

    # Build sklearn pipeline (vectorizer + classifier)
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(**TFIDF_PARAMS)),
        ("clf", LogisticRegression(**LR_PARAMS)),
    ])

    # Cross-validation on training set
    logger.info("Running 5-fold stratified cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1_weighted")
    logger.info(f"CV F1 (weighted): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Final training on full training set
    logger.info("Training final model on full training set...")
    start = time.time()
    pipeline.fit(X_train, y_train)
    train_time = time.time() - start
    logger.info(f"Training complete in {train_time:.2f}s")

    return pipeline, X_train, X_test, y_train, y_test


def evaluate_model(
    pipeline: Pipeline,
    X_test: np.ndarray,
    y_test: np.ndarray,
    le: LabelEncoder,
) -> dict:
    """Run full evaluation and return metrics dict."""
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1_weighted = f1_score(y_test, y_pred, average="weighted")
    f1_macro = f1_score(y_test, y_pred, average="macro")

    report = classification_report(
        y_test, y_pred,
        target_names=le.classes_,
        output_dict=True,
    )

    logger.info(f"\n{'='*60}")
    logger.info(f"BASELINE MODEL EVALUATION RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"Accuracy:   {accuracy:.4f}")
    logger.info(f"F1 (weighted): {f1_weighted:.4f}")
    logger.info(f"F1 (macro):    {f1_macro:.4f}")
    logger.info(f"\nPer-class metrics:")
    logger.info(f"\n{classification_report(y_test, y_pred, target_names=le.classes_)}")

    return {
        "accuracy": accuracy,
        "f1_weighted": f1_weighted,
        "f1_macro": f1_macro,
        "classification_report": report,
    }


def save_artifacts(
    pipeline: Pipeline,
    le: LabelEncoder,
    metrics: dict,
    model_dir: Path,
) -> None:
    """Save all model artifacts to disk."""
    # Save sklearn pipeline
    pipeline_path = model_dir / "pipeline.pkl"
    with open(pipeline_path, "wb") as f:
        pickle.dump(pipeline, f)
    logger.info(f"Pipeline saved: {pipeline_path}")

    # Save label encoder
    le_path = model_dir / "label_encoder.pkl"
    with open(le_path, "wb") as f:
        pickle.dump(le, f)
    logger.info(f"Label encoder saved: {le_path}")

    # Save metrics as YAML for easy inspection
    metrics_path = model_dir / "metrics.yaml"
    save_metrics = {
        "accuracy": float(metrics["accuracy"]),
        "f1_weighted": float(metrics["f1_weighted"]),
        "f1_macro": float(metrics["f1_macro"]),
        "model": "TF-IDF + Logistic Regression",
        "version": "v1.0",
    }
    with open(metrics_path, "w") as f:
        yaml.dump(save_metrics, f, default_flow_style=False)
    logger.info(f"Metrics saved: {metrics_path}")

    # Save class names for reference
    classes_path = model_dir / "classes.txt"
    with open(classes_path, "w") as f:
        f.write("\n".join(le.classes_))
    logger.info(f"Classes saved: {classes_path}")


def main() -> None:
    """Main training entrypoint."""
    logger.info("Starting baseline model training pipeline")
    logger.info("Model: TF-IDF + Logistic Regression")

    # Step 1: Load and preprocess
    df, le = load_and_preprocess_data(DATA_PATH)

    # Step 2: Train
    pipeline, X_train, X_test, y_train, y_test = train_model(df)

    # Step 3: Evaluate
    metrics = evaluate_model(pipeline, X_test, y_test, le)

    # Step 4: Save
    save_artifacts(pipeline, le, metrics, MODEL_DIR)

    logger.success("Baseline model training pipeline complete!")
    logger.info(f"Model artifacts saved to: {MODEL_DIR}")


if __name__ == "__main__":
    main()
