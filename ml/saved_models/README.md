# Saved Models

Model artifacts are generated at runtime — not stored in git.

## Generate artifacts

```bash
# Train baseline model
py ml/training/train_baseline.py

# Run MLflow experiment
py ml/experiments/mlflow_trainer.py
```

## Baseline artifacts (after training)
- pipeline.pkl — TF-IDF + Logistic Regression sklearn pipeline
- label_encoder.pkl — LabelEncoder for intent classes
- metrics.yaml — Training metrics summary
- classes.txt — List of intent class names

## Transformer artifacts (Phase 2)
- Coming soon: DistilBERT fine-tuned model
