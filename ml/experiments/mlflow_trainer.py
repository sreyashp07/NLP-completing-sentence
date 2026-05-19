"""
MLflow experiment tracking for CustomerIntent baseline model.

Logs:
- Parameters (TF-IDF config, LR config)
- Metrics (accuracy, F1, precision, recall)
- Artifacts (model pickle, confusion matrix, metrics CSV)
- Tags (model type, dataset info)

Run:
    py ml/experiments/mlflow_trainer.py
Then view UI:
    py -m mlflow ui
"""
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from ml.preprocessing.text_cleaner import TextCleaner

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_PATH  = "data/raw/customer_support_dataset.csv"
MODEL_DIR  = Path("ml/saved_models/baseline")
MLFLOW_DIR = "./mlruns"
EXPERIMENT_NAME = "customer-intent-classification"

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


def load_data():
    df = pd.read_csv(DATA_PATH)
    cleaner = TextCleaner(remove_stopwords=False, lemmatize=True)
    df["cleaned_text"] = cleaner.clean_batch(df["text"].tolist())
    le = LabelEncoder()
    df["label"] = le.fit_transform(df["intent"])
    X = df["cleaned_text"].to_numpy()
    y = df["label"].to_numpy()
    return X, y, le, df


def plot_confusion_matrix_mlflow(cm, class_names, path):
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        linewidths=0.5, linecolor="gray",
    )
    plt.title("Confusion Matrix", fontsize=14)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()


def run_experiment():
    mlflow.set_tracking_uri(MLFLOW_DIR)
    mlflow.set_experiment(EXPERIMENT_NAME)

    print(f"\nMLflow tracking URI : {MLFLOW_DIR}")
    print(f"Experiment name     : {EXPERIMENT_NAME}\n")

    X, y, le, df = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(**TFIDF_PARAMS)),
        ("clf",   LogisticRegression(**LR_PARAMS)),
    ])

    with mlflow.start_run(run_name="tfidf-lr-baseline-v1") as run:
        print(f"Run ID: {run.info.run_id}")

        # ── Log tags ──────────────────────────────────────────────────────────
        mlflow.set_tags({
            "model_type":     "TF-IDF + Logistic Regression",
            "dataset":        "synthetic_customer_support",
            "num_classes":    str(len(le.classes_)),
            "train_samples":  str(len(X_train)),
            "test_samples":   str(len(X_test)),
            "engineer":       "sreyashp07",
        })

        # ── Log TF-IDF params ─────────────────────────────────────────────────
        mlflow.log_param("tfidf_max_features",  TFIDF_PARAMS["max_features"])
        mlflow.log_param("tfidf_ngram_min",     TFIDF_PARAMS["ngram_range"][0])
        mlflow.log_param("tfidf_ngram_max",     TFIDF_PARAMS["ngram_range"][1])
        mlflow.log_param("tfidf_sublinear_tf",  TFIDF_PARAMS["sublinear_tf"])
        mlflow.log_param("tfidf_min_df",        TFIDF_PARAMS["min_df"])
        mlflow.log_param("tfidf_max_df",        TFIDF_PARAMS["max_df"])

        # ── Log LR params ─────────────────────────────────────────────────────
        mlflow.log_param("lr_C",            LR_PARAMS["C"])
        mlflow.log_param("lr_max_iter",     LR_PARAMS["max_iter"])
        mlflow.log_param("lr_solver",       LR_PARAMS["solver"])
        mlflow.log_param("lr_class_weight", LR_PARAMS["class_weight"])
        mlflow.log_param("random_state",    LR_PARAMS["random_state"])
        mlflow.log_param("test_size",       0.2)

        # ── Cross-validation ──────────────────────────────────────────────────
        print("Running 5-fold cross-validation...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(
            pipeline, X_train, y_train, cv=cv, scoring="f1_weighted"
        )
        mlflow.log_metric("cv_f1_mean", float(cv_scores.mean()))
        mlflow.log_metric("cv_f1_std",  float(cv_scores.std()))
        print(f"CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # ── Train ─────────────────────────────────────────────────────────────
        print("Training final model...")
        pipeline.fit(X_train, y_train)

        # ── Evaluate ──────────────────────────────────────────────────────────
        y_pred = pipeline.predict(X_test)

        acc       = accuracy_score(y_test, y_pred)
        f1_w      = f1_score(y_test, y_pred, average="weighted")
        f1_m      = f1_score(y_test, y_pred, average="macro")
        precision = precision_score(y_test, y_pred, average="weighted")
        recall    = recall_score(y_test, y_pred, average="weighted")

        mlflow.log_metric("accuracy",           acc)
        mlflow.log_metric("f1_weighted",        f1_w)
        mlflow.log_metric("f1_macro",           f1_m)
        mlflow.log_metric("precision_weighted", precision)
        mlflow.log_metric("recall_weighted",    recall)

        print(f"\nMetrics logged:")
        print(f"  Accuracy  : {acc:.4f}")
        print(f"  F1 (w)    : {f1_w:.4f}")
        print(f"  F1 (macro): {f1_m:.4f}")

        # ── Log confusion matrix as artifact ──────────────────────────────────
        tmp_dir = Path("ml/experiments/tmp")
        tmp_dir.mkdir(parents=True, exist_ok=True)

        cm = confusion_matrix(y_test, y_pred)
        cm_path = tmp_dir / "confusion_matrix.png"
        plot_confusion_matrix_mlflow(cm, le.classes_, cm_path)
        mlflow.log_artifact(str(cm_path), artifact_path="plots")

        # ── Log per-class metrics as CSV ──────────────────────────────────────
        report = classification_report(
            y_test, y_pred, target_names=le.classes_, output_dict=True
        )
        rows = []
        for cls in le.classes_:
            rows.append({
                "class":     cls,
                "precision": report[cls]["precision"],
                "recall":    report[cls]["recall"],
                "f1-score":  report[cls]["f1-score"],
                "support":   int(report[cls]["support"]),
            })
        report_df = pd.DataFrame(rows)
        csv_path = tmp_dir / "per_class_metrics.csv"
        report_df.to_csv(csv_path, index=False)
        mlflow.log_artifact(str(csv_path), artifact_path="metrics")

        # ── Log model ─────────────────────────────────────────────────────────
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            registered_model_name="CustomerIntent-Baseline",
        )

        print(f"\nMLflow run complete!")
        print(f"Run ID   : {run.info.run_id}")
        print(f"\nView UI  : py -m mlflow ui")
        print(f"Then open: http://localhost:5000")


if __name__ == "__main__":
    run_experiment()
