"""
Confidence score analysis across all intent classes.
Helps understand model certainty distribution.

Run: py notebooks/05_confidence_analysis.py
"""
import sys
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ml.preprocessing.text_cleaner import TextCleaner

MODEL_DIR = Path("ml/saved_models/baseline")

with open(MODEL_DIR / "pipeline.pkl", "rb") as f:
    pipeline = pickle.load(f)
with open(MODEL_DIR / "label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

df = pd.read_csv("data/raw/customer_support_dataset.csv")
cleaner = TextCleaner(remove_stopwords=False, lemmatize=True)
df["cleaned"] = cleaner.clean_batch(df["text"].tolist())
df["label"] = le.transform(df["intent"])

X = df["cleaned"].to_numpy()
y = df["label"].to_numpy()

_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

probs = pipeline.predict_proba(X_test)
max_confs = probs.max(axis=1)

print("\n=== CONFIDENCE SCORE ANALYSIS ===\n")
print(f"Mean confidence  : {max_confs.mean():.4f}")
print(f"Median confidence: {np.median(max_confs):.4f}")
print(f"Min confidence   : {max_confs.min():.4f}")
print(f"Max confidence   : {max_confs.max():.4f}")
print(f"\nConfidence buckets:")
print(f"  >= 0.90 : {(max_confs >= 0.90).sum()} samples ({(max_confs >= 0.90).mean()*100:.1f}%)")
print(f"  >= 0.75 : {(max_confs >= 0.75).sum()} samples ({(max_confs >= 0.75).mean()*100:.1f}%)")
print(f"  >= 0.60 : {(max_confs >= 0.60).sum()} samples ({(max_confs >= 0.60).mean()*100:.1f}%)")
print(f"  <  0.60 : {(max_confs < 0.60).sum()} samples ({(max_confs < 0.60).mean()*100:.1f}%)")
