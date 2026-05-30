"""
Baseline model training: TF-IDF + Logistic Regression.
Production training script with full evaluation pipeline.

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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from ml.preprocessing.text_cleaner import TextCleaner

DATA_PATH = "data/raw/customer_support_dataset.csv"
MODEL_DIR = Path("ml/saved_models/baseline")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TFIDF_PARAMS = {
    "max_features": 15000,
    "ngram_range": (1, 3),
    "sublinear_tf": True,
    "min_df": 2,
    "max_df": 0.95,
    "analyzer": "word",
    "strip_accents": "unicode",
}

LR_PARAMS = {
    "C": 5.0,
    "max_iter": 1000,
    "class_weight": "balanced",
    "solver": "lbfgs",
    "random_state": 42,
}


def load_and_preprocess_data(data_path: str) -> tuple:
    print(f"Loading dataset from {data_path}")
    df = pd.read_csv(data_path)
    print(f"Dataset: {len(df)} samples, {df['intent'].nunique()} classes")

    cleaner = TextCleaner(
        remove_urls=True, remove_emails=True, handle_emojis=True,
        expand_contractions=True, lowercase=True, remove_special_chars=True,
        remove_stopwords=False, lemmatize=True,
    )
    print("Running NLP preprocessing...")
    start = time.time()
    df["cleaned_text"] = cleaner.clean_batch(df["text"].tolist())
    print(f"Preprocessing done: {time.time()-start:.2f}s")

    le = LabelEncoder()
    df["label"] = le.fit_transform(df["intent"])
    return df, le


def train_model(df: pd.DataFrame) -> tuple:
    X = df["cleaned_text"].to_numpy()
    y = df["label"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(**TFIDF_PARAMS)),
        ("clf", LogisticRegression(**LR_PARAMS)),
    ])

    print("Running 5-fold cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1_weighted")
    print(f"CV F1: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    print("Training final model...")
    pipeline.fit(X_train, y_train)
    return pipeline, X_train, X_test, y_train, y_test


def evaluate_model(pipeline, X_test, y_test, le) -> dict:
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1_w = f1_score(y_test, y_pred, average="weighted")
    f1_m = f1_score(y_test, y_pred, average="macro")
    print(f"\nAccuracy:    {accuracy:.4f}")
    print(f"F1 weighted: {f1_w:.4f}")
    print(f"F1 macro:    {f1_m:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=le.classes_)}")
    return {"accuracy": accuracy, "f1_weighted": f1_w, "f1_macro": f1_m}


def save_artifacts(pipeline, le, metrics, model_dir: Path) -> None:
    with open(model_dir / "pipeline.pkl", "wb") as f:
        pickle.dump(pipeline, f)
    with open(model_dir / "label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)
    with open(model_dir / "metrics.yaml", "w") as f:
        yaml.dump({
            "accuracy": float(metrics["accuracy"]),
            "f1_weighted": float(metrics["f1_weighted"]),
            "f1_macro": float(metrics["f1_macro"]),
            "model": "TF-IDF + Logistic Regression",
            "version": "v1.0",
        }, f)
    with open(model_dir / "classes.txt", "w") as f:
        f.write("\n".join(le.classes_))
    print(f"Artifacts saved to: {model_dir}")


def main() -> None:
    print("="*50)
    print("CustomerIntent Baseline Model Training")
    print("="*50)
    df, le = load_and_preprocess_data(DATA_PATH)
    pipeline, X_train, X_test, y_train, y_test = train_model(df)
    metrics = evaluate_model(pipeline, X_test, y_test, le)
    save_artifacts(pipeline, le, metrics, MODEL_DIR)
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
