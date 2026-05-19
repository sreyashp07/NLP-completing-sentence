"""
Model evaluation module.

Generates:
- Confusion matrix (saved as PNG)
- Per-class precision, recall, F1
- Overall metrics summary
- Misclassification analysis

Run:
    py ml/evaluation/evaluator.py
"""
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ml.preprocessing.text_cleaner import TextCleaner

MODEL_DIR = Path("ml/saved_models/baseline")
EVAL_DIR = Path("ml/evaluation/reports")
EVAL_DIR.mkdir(parents=True, exist_ok=True)
DATA_PATH = "data/raw/customer_support_dataset.csv"


def load_artifacts():
    with open(MODEL_DIR / "pipeline.pkl", "rb") as f:
        pipeline = pickle.load(f)
    with open(MODEL_DIR / "label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    return pipeline, le


def load_test_data(le):
    df = pd.read_csv(DATA_PATH)
    cleaner = TextCleaner(remove_stopwords=False, lemmatize=True)
    df["cleaned_text"] = cleaner.clean_batch(df["text"].tolist())
    df["label"] = le.transform(df["intent"])

    X = df["cleaned_text"].to_numpy()
    y = df["label"].to_numpy()

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_test, y_test


def plot_confusion_matrix(cm, class_names, output_path):
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        linecolor="gray",
    )
    plt.title("Confusion Matrix — CustomerIntent Baseline Model", fontsize=14, pad=15)
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved: {output_path}")


def plot_per_class_metrics(report_df, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics = ["precision", "recall", "f1-score"]
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    for ax, metric, color in zip(axes, metrics, colors):
        bars = ax.barh(
            report_df["class"],
            report_df[metric],
            color=color,
            alpha=0.85,
            edgecolor="white",
        )
        ax.set_xlim(0, 1.1)
        ax.set_title(metric.capitalize(), fontsize=12, fontweight="bold")
        ax.set_xlabel("Score", fontsize=10)
        ax.axvline(x=report_df[metric].mean(), color="red", linestyle="--", alpha=0.7, label="Mean")
        ax.legend(fontsize=8)
        for bar, val in zip(bars, report_df[metric]):
            ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f}", va="center", fontsize=8)

    plt.suptitle("Per-Class Metrics — CustomerIntent Baseline", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Per-class metrics saved: {output_path}")


def run_evaluation():
    print("\n" + "="*60)
    print("  CUSTOMERINTENT MODEL EVALUATION")
    print("="*60)

    pipeline, le = load_artifacts()
    X_test, y_test = load_test_data(le)
    class_names = le.classes_

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)

    accuracy  = accuracy_score(y_test, y_pred)
    f1_w      = f1_score(y_test, y_pred, average="weighted")
    f1_m      = f1_score(y_test, y_pred, average="macro")
    precision = precision_score(y_test, y_pred, average="weighted")
    recall    = recall_score(y_test, y_pred, average="weighted")

    print(f"\n Overall Metrics:")
    print(f"   Accuracy          : {accuracy:.4f}")
    print(f"   Precision (weighted): {precision:.4f}")
    print(f"   Recall (weighted)   : {recall:.4f}")
    print(f"   F1 (weighted)       : {f1_w:.4f}")
    print(f"   F1 (macro)          : {f1_m:.4f}")

    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    print(f"\n Per-Class Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, class_names, EVAL_DIR / "confusion_matrix.png")

    rows = []
    for cls in class_names:
        rows.append({
            "class": cls,
            "precision": report[cls]["precision"],
            "recall":    report[cls]["recall"],
            "f1-score":  report[cls]["f1-score"],
            "support":   int(report[cls]["support"]),
        })
    report_df = pd.DataFrame(rows)
    plot_per_class_metrics(report_df, EVAL_DIR / "per_class_metrics.png")

    report_df.to_csv(EVAL_DIR / "metrics_report.csv", index=False)
    print(f"\n CSV report saved: {EVAL_DIR}/metrics_report.csv")

    summary = {
        "accuracy":          float(accuracy),
        "precision_weighted": float(precision),
        "recall_weighted":   float(recall),
        "f1_weighted":       float(f1_w),
        "f1_macro":          float(f1_m),
        "test_samples":      int(len(y_test)),
        "num_classes":       int(len(class_names)),
        "model":             "TF-IDF + Logistic Regression",
    }
    with open(EVAL_DIR / "summary.yaml", "w") as f:
        yaml.dump(summary, f, default_flow_style=False)
    print(f" Summary YAML saved: {EVAL_DIR}/summary.yaml")

    print("\n" + "="*60)
    print("  EVALUATION COMPLETE")
    print("="*60 + "\n")


if __name__ == "__main__":
    run_evaluation()
