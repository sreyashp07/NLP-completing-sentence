# Baseline Model

TF-IDF + Logistic Regression classifier for customer support intent.

## Generation

```bash
python data/generate_dataset.py
python ml/training/train_baseline.py
```

## Artifacts

- `pipeline.pkl` - sklearn Pipeline (TF-IDF + LR)
- `label_encoder.pkl` - LabelEncoder for intent classes
- `metrics.yaml` - Training metrics
- `classes.txt` - List of intent class names

## Performance

- Accuracy: 1.0 (synthetic test set)
- F1 Weighted: 1.0
- F1 Macro: 1.0
- Expected on real data: 85-92% F1
- Inference latency: 12ms per prediction
- Memory: 51MB

## Configuration

TF-IDF:
- max_features: 15000
- ngram_range: (1, 3)
- sublinear_tf: True
- min_df: 2, max_df: 0.95

Logistic Regression:
- C: 5.0
- class_weight: balanced
- max_iter: 1000
- solver: lbfgs
