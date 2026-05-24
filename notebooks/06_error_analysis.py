"""
Detailed error analysis for misclassified samples.

Identifies patterns in wrong predictions to guide
model improvement and data collection strategy.

Run: py notebooks/06_error_analysis.py
"""
import sys
import pickle
from pathlib import Path
from collections import defaultdict

import pandas as pd
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
texts = df["text"].to_numpy()

_, X_test, _, y_test, _, texts_test = train_test_split(
    X, y, texts, test_size=0.2, random_state=42, stratify=y
)

y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)

confusion_pairs = defaultdict(int)
wrong_samples = []

for i, (true, pred) in enumerate(zip(y_test, y_pred)):
    if true != pred:
        true_label = le.classes_[true]
        pred_label = le.classes_[pred]
        confusion_pairs[(true_label, pred_label)] += 1
        wrong_samples.append({
            "text": texts_test[i],
            "true": true_label,
            "predicted": pred_label,
            "confidence": max(y_prob[i]),
        })

print("\n=== ERROR ANALYSIS REPORT ===\n")
print(f"Total test samples  : {len(y_test)}")
print(f"Misclassified       : {len(wrong_samples)}")
print(f"Error rate          : {len(wrong_samples)/len(y_test)*100:.2f}%\n")

if confusion_pairs:
    print("Most common confusion pairs:")
    sorted_pairs = sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)
    for (true, pred), count in sorted_pairs[:5]:
        print(f"  {true:25} -> {pred:25} ({count} times)")

    print("\nSample misclassified messages:")
    for s in wrong_samples[:3]:
        print(f"\n  Text      : {s['text'][:70]}")
        print(f"  True      : {s['true']}")
        print(f"  Predicted : {s['predicted']}")
        print(f"  Confidence: {s['confidence']:.3f}")
else:
    print("No misclassifications found on this dataset.")
